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
  "www.haaretz.co.il",
  "rss.walla.co.il",
  "www.mako.co.il",
  "www.kan.org.il",
  "www.globes.co.il",
  "www.israelhayom.co.il",
  "www.jpost.com",
  "www.timesofisrael.com",
  "www.themarker.com",
  "www.geektime.co.il",
  "www.tgspot.co.il",
  "www.one.co.il",
  "www.sport5.co.il",
  "feeds.apnews.com",
  "feeds.npr.org",
  "rss.dw.com",
  "www.france24.com",
  "www.aljazeera.com",
  "feeds.skynews.com",
  "www.cbc.ca",
  "www.spiegel.de",
  "www.lemonde.fr",
  "www.arabnews.com",
  "feeds.bloomberg.com",
  "www.ft.com",
  "www.cnbc.com",
  "techcrunch.com",
  "www.theverge.com",
  "feeds.arstechnica.com",
  "www.wired.com",
  "www.technologyreview.com",
  "openai.com",
  "deepmind.google",
  "www.anthropic.com",
  "www.nasa.gov",
  "www.esa.int",
  "www.space.com",
  "www.sciencedaily.com",
  "feeds.ign.com",
  "www.gamespot.com",
  "www.rollingstone.com",
  "variety.com",
  "www.reddit.com",
  "www.producthunt.com",
  "mshibanami.github.io",
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
  "hacker-news.firebaseio.com",
  "api.airplanes.live",
  "api.binance.com",
  "open.er-api.com",
  "api.github.com",
  "image.tmdb.org",
  "pixabay.com",
  "cdn.pixabay.com",
  "www.wikidata.org",
  "query.wikidata.org",
  "api.tvmaze.com",
  "archive.org",
  "sepiasearch.org",
  "peertube.cpy.re",
  "world.openfoodfacts.org",
  "images.openfoodfacts.org",
  "static.openfoodfacts.org",
  "cheapersal.co.il",
  "price-api.additlist.com",
  "huggingface.co",
  "api-inference.huggingface.co",
  "router.huggingface.co",
  ".hf.space",
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
  "time.now",
  "ipapi.co",
  "posix4e.github.io",
  "iptv-org.github.io",
  "de1.api.radio-browser.info",
  "api.radio-browser.info",
  "cdn.jsdelivr.net",
];

type RelayFn = (target: string) => string;

const buildPublicRelays = (): RelayFn[] => {
  const relays: RelayFn[] = [];
  const custom = (import.meta.env.VITE_CORS_PROXY_URL as string | undefined)?.trim();
  if (custom) {
    const base = custom.replace(/\/?$/, "/");
    relays.push((target) => `${base}${encodeURIComponent(target)}`);
    return relays;
  }
  if (isStaticWebHost()) {
    return relays;
  }
  relays.push(
    (target) => `https://corsproxy.io/?${encodeURIComponent(target)}`,
    (target) => `https://api.allorigins.win/raw?url=${encodeURIComponent(target)}`,
    (target) => `https://api.allorigins.win/get?url=${encodeURIComponent(target)}`,
    (target) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(target)}`,
    (target) => `https://r.jina.ai/${target}`,
  );
  return relays;
};

export function isStaticWebHost(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  if (hostname.endsWith("github.io")) return true;
  if (import.meta.env.DEV) return false;
  if (port === "5173" || port === "4173") return false;
  if ((hostname === "localhost" || hostname === "127.0.0.1") && port === "3000") return false;
  return hostname !== "localhost" && hostname !== "127.0.0.1";
}

/** True when the app is served as a static site (GitHub Pages, CDN) — no local dev proxy. */
export const isWebOnlyHost = (): boolean => isStaticWebHost();

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

/** Invidious/Piped never send CORS — skip direct browser fetch to avoid console noise. */
const isYouTubeRelayHost = (host: string): boolean => {
  if (!host) return false;
  if (host === "yewtu.be") return true;
  if (/^pipedapi\./i.test(host) || /^api\.piped\./i.test(host)) return true;
  if (/^invidious\./i.test(host) || /^inv\./i.test(host)) return true;
  if (host.endsWith(".tux.pizza")) return true;
  return false;
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

const buildDevProxyUrl = (target: string): string =>
  `/api/proxy?url=${encodeURIComponent(target)}`;

const relayUsesAllOriginsGet = (relayUrl: string): boolean =>
  relayUrl.includes("allorigins.win/get");

const bodyFromRelayResponse = async (relayUrl: string, response: Response): Promise<string> => {
  if (relayUsesAllOriginsGet(relayUrl)) {
    const json = (await response.json()) as { contents?: string; status?: { http_code?: number } };
    const code = json.status?.http_code ?? 0;
    if (code >= 400 || !json.contents?.trim()) {
      throw new Error(`relay empty (HTTP ${code || response.status})`);
    }
    return json.contents;
  }
  const text = await response.text();
  if (!text.trim()) throw new Error("relay empty body");
  return text;
};

async function fetchViaRelays(url: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {
    Accept: "application/json, application/xml, text/plain, application/rss+xml, application/atom+xml, */*",
    ...(init?.headers as Record<string, string> | undefined),
  };
  const signal = init?.signal;
  const relays = buildPublicRelays();
  if (!relays.length) {
    throw new Error(`No CORS proxy configured for static host (${url})`);
  }

  const tryRelay = async (relay: RelayFn): Promise<Response> => {
    const relayUrl = relay(url);
    const r = await fetch(relayUrl, { method: "GET", headers, signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await bodyFromRelayResponse(relayUrl, r);
    return new Response(body, { status: 200, headers: { "Content-Type": "text/xml" } });
  };

  if (isStaticWebHost()) {
    let lastErr: unknown;
    for (const relay of relays) {
      try {
        return await tryRelay(relay);
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error(`All CORS relays failed for ${url}`);
  }

  const attempts = relays.map((relay) => tryRelay(relay));
  try {
    return await Promise.any(attempts);
  } catch {
    const settled = await Promise.allSettled(attempts);
    const firstErr = settled.find((s) => s.status === "rejected") as PromiseRejectedResult | undefined;
    throw firstErr?.reason instanceof Error ? firstErr.reason : new Error(`All CORS relays failed for ${url}`);
  }
}

export async function proxyAwareFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const inBrowser = typeof window !== "undefined";
  const method = init?.method ?? "GET";
  const host = hostOf(url);

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

  if (method === "GET" && isYouTubeRelayHost(host)) {
    if (import.meta.env.DEV) {
      try {
        const r = await fetch(buildDevProxyUrl(url), { ...init, method: "GET" });
        if (r.ok) return r;
      } catch {
        /* no direct fallback for relay-only hosts */
      }
    }
    if (isStaticWebHost()) {
      try {
        return await fetchViaRelays(url, init);
      } catch {
        throw new Error(`YouTube relay fetch failed (${host})`);
      }
    }
    throw new Error(`YouTube relay unavailable (${host})`);
  }

  if (import.meta.env.DEV && method === "POST") {
    return fetch(buildDevProxyUrl(url), init);
  }

  // Static web (GitHub Pages): RSS/news hosts never send CORS — go straight to relays.
  if (isStaticWebHost() && method === "GET" && PROXY_REQUIRED.has(host)) {
    try {
      return await fetchViaRelays(url, init);
    } catch {
      /* fall through to direct last-resort */
    }
  }

  const preferDirect = (hasDirectCors(url) || !needsProxy(url)) && !isYouTubeRelayHost(host);
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
  return isStaticWebHost() ? 26_000 : 9_000;
}
