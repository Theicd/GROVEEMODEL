/** Shared online probe — used by UI hooks and news fetch policy. */

export type NetworkReachability = "online" | "offline" | "limited";

const PROBE_TIMEOUT_MS = 4_000;

export async function probeNetworkReachable(): Promise<boolean> {
  if (typeof navigator !== "undefined" && !navigator.onLine) return false;
  try {
    const base = import.meta.env.BASE_URL || "/";
    const url = `${base}${base.endsWith("/") ? "" : "/"}favicon.ico?_=${Date.now()}`;
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), PROBE_TIMEOUT_MS);
    await fetch(url, { method: "HEAD", cache: "no-store", signal: ctrl.signal });
    window.clearTimeout(timer);
    return true;
  } catch {
    return false;
  }
}

export async function resolveNetworkReachability(): Promise<NetworkReachability> {
  if (typeof navigator !== "undefined" && !navigator.onLine) return "offline";
  const ok = await probeNetworkReachable();
  return ok ? "online" : "limited";
}
