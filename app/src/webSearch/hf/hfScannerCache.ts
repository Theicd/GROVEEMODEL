import type { ScannerModelRow } from "./hfApiScannerClient";

const DB_NAME = "grovee-hf-scanner";
const DB_VERSION = 1;
const STORE = "working-models";
const CACHE_KEY = "latest";
const CACHE_TTL_MS = 6 * 60 * 60_000;

type CacheRecord = {
  key: string;
  rows: ScannerModelRow[];
  fetchedAt: number;
};

let memoryCache: CacheRecord | null = null;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("indexedDB unavailable"));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("indexedDB open failed"));
  });
}

function withStore<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const store = tx.objectStore(STORE);
        const req = fn(store);
        let result: T | undefined;
        req.onsuccess = () => {
          result = req.result as T;
        };
        req.onerror = () => {
          db.close();
          reject(req.error ?? new Error("indexedDB request failed"));
        };
        tx.oncomplete = () => {
          db.close();
          resolve(result as T);
        };
        tx.onerror = () => {
          db.close();
          reject(tx.error ?? new Error("indexedDB tx failed"));
        };
      }),
  );
}

function readMemoryCache(maxAgeMs: number): ScannerModelRow[] | null {
  if (!memoryCache?.rows?.length) return null;
  if (Date.now() - memoryCache.fetchedAt > maxAgeMs) return null;
  return memoryCache.rows;
}

export async function readWorkingModelsCache(maxAgeMs = CACHE_TTL_MS): Promise<ScannerModelRow[] | null> {
  const fromMemory = readMemoryCache(maxAgeMs);
  if (fromMemory) return fromMemory;
  try {
    const record = await withStore<CacheRecord | undefined>("readonly", (store) => store.get(CACHE_KEY));
    if (!record?.rows?.length) return null;
    if (Date.now() - record.fetchedAt > maxAgeMs) return null;
    memoryCache = record;
    return record.rows;
  } catch {
    return readMemoryCache(maxAgeMs);
  }
}

export async function writeWorkingModelsCache(rows: ScannerModelRow[]): Promise<void> {
  if (!rows.length) return;
  const record: CacheRecord = { key: CACHE_KEY, rows, fetchedAt: Date.now() };
  memoryCache = record;
  try {
    await withStore<IDBValidKey>("readwrite", (store) => store.put(record));
  } catch {
    /* memory cache still holds the snapshot */
  }
}

/** Reset cache (tests). */
export async function clearWorkingModelsCache(): Promise<void> {
  memoryCache = null;
  try {
    await withStore<undefined>("readwrite", (store) => store.delete(CACHE_KEY));
  } catch {
    /* ignore */
  }
}
