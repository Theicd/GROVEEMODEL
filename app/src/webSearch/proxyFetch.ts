/** Browser-safe fetch — dev proxy, direct CORS where allowed, parallel relays on static hosts. */

const PROXY_REQUIRED = new Set([
  "overpass-api.de",
  "nominatim.openstreetmap.org",
  "router.project-osrm.org",
  "opensky-network.org",
  "www.oref.org.il",
  "api.tzevaadom.co.il",
  "feeds.bbci.co.uk",
  "rss.cnn.com",
  "feeds.reuters.com",
  "www.theguardian.com",
  "www.ynet.co.il",
  "stooq.com",
  "query1.finance.yahoo.com",
  "finance.yahoo.com",
]);

/** These APIs allow browser CORS — prefer direct fetch (fast, reliable on GitHub Pages). */
const CORS_DIRECT_SUFFIXES = [
  ".wikipedia.org",
  "api.open-meteo.com",
  "geocoding-api.open-meteo.com",
  "marine-api.open-meteo.com",
  "air-quality-api.open-meteo.com",
  "export.arxiv.org",
  "api.airplanes.live",
  "api.binance.com",
  "open.er-api.com",
  "api.github.com",
  "huggingface.co",
  "api-inference.huggingface.co",
  "meri.digitraffic.fi",
  "celestrak.org",
  "ll.thespacedevs.com",
  "timeapi.io",
  "api.wheretheiss.at",
  "earthquake.usgs.gov",
  "date.nager.at",
  "query.wikidata.org",
  "www.wikidata.org",
  "eonet.gsfc.nasa.gov",
  "www.gdacs.org",
  "api.open-notify.org",
  "api.open-notify.org",
  "time.now",
  "ipapi.co",
];

export function isStaticWebHost(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  if (hostname.endsWith("github.io")) return true;
  if (import.meta.env.DEV) return false;
  if (port === "5173" || port === "4173") return false;
  if ((hostname === "localhost" || hostname === "127.0.0.1") && port === "3000") return false;
  return hostname !== "localhost" && hostname !== "127.0.0.1";
}

const isCrossOrigin = (url: string): boolean => {
  if (typeof window === "undefined") return true;
  try {
    return new URL(url).origin !== window.location.origin;
  } catch {
    return true;
  }
};

const hostOf = (url: string): string => {
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
};

export const hasDirectCors = (url: string): boolean => {
  const host = hostOf(url);
  if (!host) return false;
  return CORS_DIRECT_SUFFIXES.some((s) => host === s || host.endsWith(s));
};

export const needsProxy = (url: string): boolean => {
  const host = hostOf(url);
  if (!host) return isStaticWebHost();
  if (hasDirectCors(url)) return false;
  if (PROXY_REQUIRED.has(host)) return true;
  return isStaticWebHost() && isCrossOrigin(url);
};

const PUBLIC_RELAYS = [
  (target: string) => `https://corsproxy.io/?${encodeURIComponent(target)}`,
  (target: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(target)}`,
  (target: string) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(target)}`,
];

const buildDevProxyUrl = (target: string): string =>
  `/api/proxy?url=${encodeURIComponent(target)}`;

async function fetchViaRelays(url: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {
    Accept: "application/json, application/xml, text/plain, */*",
    ...(init?.headers as Record<string, string> | undefined),
  };
  const signal = init?.signal;

  const tryRelay = async (relay: (t: string) => string): Promise<Response> => {
    const r = await fetch(relay(url), { method: "GET", headers, signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r;
  };

  let lastErr: Error = new Error(`All CORS relays failed for ${url}`);
  for (const relay of PUBLIC_RELAYS) {
    try {
      return await tryRelay(relay);
    } catch (err) {
      lastErr = err instanceof Error ? err : lastErr;
    }
  }
  throw lastErr;
}

export async function proxyAwareFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const inBrowser = typeof window !== "undefined";
  const method = init?.method ?? "GET";

  if (!inBrowser) {
    return fetch(url, init);
  }

  if (import.meta.env.DEV && method === "GET") {
    try {
      const r = await fetch(buildDevProxyUrl(url), { ...init, method: "GET" });
      if (r.ok) return r;
    } catch {
      /* fall through */
    }
  }

  if (import.meta.env.DEV && method === "POST") {
    return fetch(buildDevProxyUrl(url), init);
  }

  const preferDirect = hasDirectCors(url) || !needsProxy(url);
  let directFailed = false;
  if (preferDirect) {
    try {
      const r = await fetch(url, init);
      if (r.ok) return r;
      directFailed = true;
    } catch {
      directFailed = true;
    }
  }

  if (method !== "GET") {
    return fetch(url, init);
  }

  const crossOrigin = isCrossOrigin(url);
  const shouldRelay =
    needsProxy(url) ||
    (directFailed && crossOrigin && (isStaticWebHost() || !preferDirect));

  if (shouldRelay) {
    try {
      return await fetchViaRelays(url, init);
    } catch {
      /* last-resort direct below */
    }
  }

  return fetch(url, init);
}

export function defaultFetchTimeoutMs(): number {
  return isStaticWebHost() ? 22_000 : 9_000;
}
