/**
 * Helpers used by the in-app cache-clear flow.
 *
 * Lives outside App.tsx so the byte formatter and the persist-storage call
 * can be unit tested directly (App.tsx is too React-heavy to test cheaply).
 */

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "?";
  if (bytes === 0) return "0 MB";
  const KB = 1024;
  const MB = KB * 1024;
  const GB = MB * 1024;
  if (bytes >= GB) return `${(bytes / GB).toFixed(2)} GB`;
  if (bytes >= MB) return `${(bytes / MB).toFixed(0)} MB`;
  if (bytes >= KB) return `${(bytes / KB).toFixed(0)} KB`;
  return `${bytes} B`;
}

/**
 * Ask the browser to mark our origin as "persistent" so it does not auto-evict
 * gigabytes of model weights under disk pressure. No-op if unsupported, already
 * persistent, or denied (Brave / private mode often deny). Returns the granted
 * state so callers can surface a hint.
 */
export async function requestPersistentStorage(): Promise<{
  supported: boolean;
  persistent: boolean;
  reason?: string;
}> {
  try {
    const s = (typeof navigator !== "undefined" ? navigator.storage : undefined) as
      | (StorageManager & { persisted?: () => Promise<boolean>; persist?: () => Promise<boolean> })
      | undefined;
    if (!s?.persist || !s.persisted) return { supported: false, persistent: false };
    const already = await s.persisted();
    if (already) return { supported: true, persistent: true };
    const granted = await s.persist();
    return { supported: true, persistent: granted, reason: granted ? undefined : "denied" };
  } catch (e) {
    return { supported: true, persistent: false, reason: e instanceof Error ? e.message : String(e) };
  }
}
