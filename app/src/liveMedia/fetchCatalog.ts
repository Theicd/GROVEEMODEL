import { proxyAwareFetch } from "../webSearch/proxyFetch";

export async function fetchCatalogText(url: string, timeoutMs = 22_000): Promise<string> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await proxyAwareFetch(url, {
      signal: ctrl.signal,
      headers: { Accept: "*/*" },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.text();
  } finally {
    clearTimeout(timer);
  }
}
