/** Browser-safe fetch — direct first, then dev proxy / public CORS relay. */

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
]);

const needsProxy = (url: string): boolean => {
  try {
    const host = new URL(url).hostname;
    return PROXY_HOSTS.has(host);
  } catch {
    return false;
  }
};

const buildProxyUrl = (target: string): string => {
  if (import.meta.env.DEV) {
    return `/api/proxy?url=${encodeURIComponent(target)}`;
  }
  return `https://api.allorigins.win/raw?url=${encodeURIComponent(target)}`;
};

export async function proxyAwareFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const inBrowser = typeof window !== "undefined";
  const forceProxy = inBrowser && needsProxy(url);

  if (!forceProxy) {
    try {
      const r = await fetch(url, init);
      if (r.ok) return r;
    } catch {
      /* retry via proxy in browser */
    }
  }

  if (!inBrowser) {
    return fetch(url, init);
  }

  const proxyUrl = buildProxyUrl(url);
  const method = init?.method ?? "GET";
  if (method === "POST" && import.meta.env.DEV) {
    return fetch(proxyUrl, init);
  }
  if (method !== "GET") {
    return fetch(url, init);
  }
  return fetch(proxyUrl, {
    ...init,
    method: "GET",
    headers: {
      Accept: "application/json, text/plain, */*",
      ...init?.headers,
    },
  });
}
