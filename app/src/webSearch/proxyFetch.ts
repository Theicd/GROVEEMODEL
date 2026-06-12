/** Browser-safe fetch — dev proxy, multi-relay on GitHub Pages / static hosts. */

const PROXY_HOSTS = new Set([
  "nominatim.openstreetmap.org",
  "router.project-osrm.org",
  "countriesnow.space",
  "www.wikidata.org",
  "query.wikidata.org",
  "opensky-network.org",
  "www.oref.org.il",
  "api.tzevaadom.co.il",
  "date.nager.at",
  "feeds.bbci.co.uk",
  "rss.cnn.com",
  "earthquake.usgs.gov",
  "api.open-meteo.com",
  "services.swpc.noaa.gov",
  "www.gdacs.org",
  "www.seismicportal.eu",
  "api.weather.gov",
  "eonet.gsfc.nasa.gov",
  "meri.digitraffic.fi",
  "wttr.in",
  "api.wheretheiss.at",
  "celestrak.org",
  "airplanes.live",
]);

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

const needsProxy = (url: string): boolean => {
  try {
    const host = new URL(url).hostname;
    if (PROXY_HOSTS.has(host)) return true;
    if (isStaticWebHost() && isCrossOrigin(url)) return true;
    return false;
  } catch {
    return isStaticWebHost();
  }
};

const PUBLIC_RELAYS = [
  (target: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(target)}`,
  (target: string) => `https://corsproxy.io/?${encodeURIComponent(target)}`,
  (target: string) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(target)}`,
];

const buildDevProxyUrl = (target: string): string =>
  `/api/proxy?url=${encodeURIComponent(target)}`;

async function fetchViaRelays(url: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {
    Accept: "application/json, application/xml, text/plain, */*",
    ...(init?.headers as Record<string, string> | undefined),
  };
  let lastErr: unknown;
  for (const relay of PUBLIC_RELAYS) {
    try {
      const r = await fetch(relay(url), {
        ...init,
        method: "GET",
        headers,
      });
      if (r.ok) return r;
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(`All CORS relays failed for ${url}`);
}

export async function proxyAwareFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const inBrowser = typeof window !== "undefined";
  const method = init?.method ?? "GET";
  const forceProxy = inBrowser && needsProxy(url);

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

  if (!forceProxy) {
    try {
      const r = await fetch(url, init);
      if (r.ok) return r;
    } catch {
      /* retry via relay */
    }
  }

  if (method !== "GET") {
    return fetch(url, init);
  }

  return fetchViaRelays(url, init);
}
