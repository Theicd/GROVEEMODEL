/**
 * Read-only snapshot of every storage area the app uses (Cache Storage,
 * IndexedDB, OPFS, Service Workers) plus storage quota totals. The Settings
 * UI uses this to show the user exactly what is on their disk *before* and
 * *after* "Clear cache", so there is no longer any guesswork about whether
 * the clear actually freed bytes.
 */

import { formatBytes } from "./storageReport";

export interface StorageAudit {
  /** Bytes the browser estimates this origin currently uses. */
  usageBytes: number;
  /** Quota the browser is willing to give this origin (best-effort or persistent). */
  quotaBytes: number;
  /** Whether the origin is marked persistent (Brave often denies; affects auto-eviction). */
  persisted: boolean;
  /** Cache Storage entries by name + how many requests each cache stores. */
  caches: Array<{ name: string; entryCount: number }>;
  /** Names of every IndexedDB database we could enumerate. `databases()` is unavailable in some browsers. */
  idbNames: string[];
  /** Top-level OPFS entries (files + folders) under our origin. */
  opfsEntries: string[];
  /** Active service worker registrations on this origin. */
  serviceWorkerCount: number;
  /** Set when the API itself failed (private mode, etc.). */
  errors: string[];
}

const safeAsync = async <T>(label: string, errors: string[], fn: () => Promise<T>, fallback: T): Promise<T> => {
  try {
    return await fn();
  } catch (e) {
    errors.push(`${label}: ${e instanceof Error ? e.message : String(e)}`);
    return fallback;
  }
};

export async function auditAppStorage(): Promise<StorageAudit> {
  const errors: string[] = [];

  let usageBytes = -1;
  let quotaBytes = -1;
  let persisted = false;
  await safeAsync("estimate", errors, async () => {
    if (navigator.storage?.estimate) {
      const e = await navigator.storage.estimate();
      usageBytes = typeof e.usage === "number" ? e.usage : -1;
      quotaBytes = typeof e.quota === "number" ? e.quota : -1;
    }
  }, undefined);
  await safeAsync("persisted", errors, async () => {
    if (navigator.storage?.persisted) persisted = await navigator.storage.persisted();
  }, undefined);

  const cachesOut: StorageAudit["caches"] = [];
  await safeAsync("caches", errors, async () => {
    if (typeof caches === "undefined") return;
    const names = await caches.keys();
    for (const name of names) {
      let entryCount = 0;
      try {
        const c = await caches.open(name);
        const reqs = await c.keys();
        entryCount = reqs.length;
      } catch {
        // ignore — still report the name
      }
      cachesOut.push({ name, entryCount });
    }
  }, undefined);

  let idbNames: string[] = [];
  await safeAsync("idb", errors, async () => {
    const idb = indexedDB as IDBFactory & { databases?: () => Promise<Array<{ name?: string }>> };
    if (!idb.databases) {
      errors.push("idb: indexedDB.databases() not supported");
      return;
    }
    const dbs = await idb.databases();
    idbNames = dbs.map((d) => d.name ?? "").filter(Boolean) as string[];
  }, undefined);

  const opfsEntries: string[] = [];
  await safeAsync("opfs", errors, async () => {
    const s = navigator.storage as StorageManager & {
      getDirectory?: () => Promise<FileSystemDirectoryHandle>;
    };
    if (!s.getDirectory) return;
    const root = await s.getDirectory();
    const walker = root as FileSystemDirectoryHandle & {
      keys?: () => AsyncIterableIterator<string>;
      entries?: () => AsyncIterableIterator<[string, FileSystemHandle]>;
    };
    if (walker.keys) {
      for await (const name of walker.keys()) opfsEntries.push(name);
    } else if (walker.entries) {
      for await (const [name] of walker.entries()) opfsEntries.push(name);
    }
  }, undefined);

  let serviceWorkerCount = 0;
  await safeAsync("serviceWorker", errors, async () => {
    if (!("serviceWorker" in navigator)) return;
    const regs = await navigator.serviceWorker.getRegistrations();
    serviceWorkerCount = regs.length;
  }, undefined);

  return {
    usageBytes,
    quotaBytes,
    persisted,
    caches: cachesOut,
    idbNames,
    opfsEntries,
    serviceWorkerCount,
    errors,
  };
}

/** "3.5 GB / 12 GB · persisted: כן" */
export function summarizeStorageHeader(audit: StorageAudit): string {
  const used = audit.usageBytes >= 0 ? formatBytes(audit.usageBytes) : "?";
  const quota = audit.quotaBytes >= 0 ? formatBytes(audit.quotaBytes) : "?";
  const persisted = audit.persisted ? "כן" : "לא";
  return `${used} / ${quota} · persisted: ${persisted}`;
}

/** Compact one-line summary used inline on the audit panel. */
export function summarizeStorageInventory(audit: StorageAudit): string {
  return [
    `${audit.caches.length} caches`,
    `${audit.idbNames.length} IDB`,
    `${audit.opfsEntries.length} OPFS`,
    `${audit.serviceWorkerCount} SW`,
  ].join(" · ");
}

export const isAuditEmpty = (a: StorageAudit): boolean =>
  a.caches.length === 0 &&
  a.idbNames.length === 0 &&
  a.opfsEntries.length === 0 &&
  a.serviceWorkerCount === 0 &&
  (a.usageBytes === 0 || a.usageBytes === -1);
