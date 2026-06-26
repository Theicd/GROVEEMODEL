import { gunzipSync } from "fflate";
import { isStaticWebHost } from "../../webSearch/proxyFetch";
import { parsedChannelsForXml, resetParsedChannelCacheForTests } from "./xmltvParse";

export type MjhEpgSource = {
  key: string;
  label: string;
  url: string;
  /** When stream URL hints this platform, try this source soon after the global index. */
  streamHint?: RegExp;
};

export const MJH_EPG_SOURCES: MjhEpgSource[] = [
  {
    key: "mjh-all",
    label: "MJH All",
    url: "https://i.mjh.nz/all/epg.xml.gz",
  },
  {
    key: "mjh-plex-us",
    label: "Plex US",
    url: "https://i.mjh.nz/Plex/us.xml.gz",
    streamHint: /plex\.(tv|wurl)|amagi\.tv|wurl\.tv|wurl\.com|xumo|mediatailor/i,
  },
  {
    key: "mjh-pluto-us",
    label: "Pluto TV US",
    url: "https://i.mjh.nz/PlutoTV/us.xml.gz",
    streamHint: /pluto\.tv/i,
  },
  {
    key: "mjh-samsung-us",
    label: "Samsung TV+ US",
    url: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz",
    streamHint: /samsung|samsungtvplus/i,
  },
  {
    key: "mjh-roku",
    label: "Roku",
    url: "https://i.mjh.nz/Roku/all.xml.gz",
    streamHint: /roku|wurl\.com/i,
  },
];

const xmlCache = new Map<string, Promise<string | null>>();

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

const MJH_CDN_BASE = "https://cdn.jsdelivr.net/gh/matthuisman/i.mjh.nz@master";

function devEpgProxyUrl(target: string): string {
  return `/api/epg/raw?url=${encodeURIComponent(target)}`;
}

/** i.mjh.nz redirects to raw.githubusercontent.com (no browser CORS) — jsDelivr mirrors the same repo with CORS. */
export function mjhUrlToCdn(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.hostname !== "i.mjh.nz" || parsed.protocol !== "https:") return null;
    const path = parsed.pathname.replace(/^\//, "");
    return path ? `${MJH_CDN_BASE}/${path}` : null;
  } catch {
    return null;
  }
}

async function fetchBytesFromUrl(fetchUrl: string): Promise<ArrayBuffer | null> {
  const res = await fetch(fetchUrl, { cache: "no-store", mode: "cors" });
  if (!res.ok) return null;
  const buf = await res.arrayBuffer();
  return buf.byteLength > 64 ? buf : null;
}

async function gunzipToText(buf: ArrayBuffer): Promise<string> {
  if (typeof DecompressionStream !== "undefined") {
    try {
      const ds = new DecompressionStream("gzip");
      const body = new Response(buf).body;
      if (body) {
        const decompressed = body.pipeThrough(ds);
        return await new Response(decompressed).text();
      }
    } catch {
      /* fall through */
    }
  }
  try {
    return new TextDecoder().decode(gunzipSync(new Uint8Array(buf)));
  } catch {
    /* fall through */
  }
  if (typeof process !== "undefined" && process.versions?.node) {
    const { gunzipSync: nodeGunzip } = await import("node:zlib");
    return nodeGunzip(Buffer.from(buf)).toString("utf8");
  }
  throw new Error("gzip decompression unavailable");
}

/** Fetch MJH gzip/XML bytes — dev proxy (binary-safe); static hosts use jsDelivr CDN (CORS-safe). */
export async function fetchMjhBytes(url: string): Promise<ArrayBuffer | null> {
  const attempts: Array<() => Promise<ArrayBuffer | null>> = [];
  const cdnUrl = mjhUrlToCdn(url);
  const staticHost = isBrowser() && isStaticWebHost();

  if (cdnUrl && staticHost) {
    attempts.push(() => fetchBytesFromUrl(cdnUrl));
  }

  if (isBrowser() && import.meta.env.DEV) {
    attempts.push(() => fetchBytesFromUrl(devEpgProxyUrl(url)));
  }

  if (!staticHost) {
    attempts.push(() => fetchBytesFromUrl(url));
  }

  if (cdnUrl && !staticHost) {
    attempts.push(() => fetchBytesFromUrl(cdnUrl));
  }

  if (staticHost) {
    attempts.push(() => fetchBytesFromUrl(`https://corsproxy.io/?${encodeURIComponent(url)}`));
  }

  for (const attempt of attempts) {
    try {
      const buf = await attempt();
      if (buf) return buf;
    } catch {
      /* try next */
    }
  }
  return null;
}

export async function fetchMjhXmltv(url: string): Promise<string | null> {
  let pending = xmlCache.get(url);
  if (!pending) {
    pending = (async () => {
      try {
        const buf = await fetchMjhBytes(url);
        if (!buf) return null;
        if (url.endsWith(".gz")) {
          const text = await gunzipToText(buf);
          return text.length > 1000 ? text : null;
        }
        const text = new TextDecoder().decode(buf);
        return text.length > 1000 ? text : null;
      } catch {
        return null;
      }
    })();
    xmlCache.set(url, pending);
  }
  return pending;
}

/** Global MJH index first, then platform-specific feeds for the stream URL. */
export function orderedSourcesForStream(streamUrl: string): MjhEpgSource[] {
  const all = MJH_EPG_SOURCES.find((s) => s.key === "mjh-all");
  const hinted = MJH_EPG_SOURCES.filter((s) => s.key !== "mjh-all" && s.streamHint?.test(streamUrl));
  const rest = MJH_EPG_SOURCES.filter((s) => s.key !== "mjh-all" && !s.streamHint?.test(streamUrl));
  return [...(all ? [all] : []), ...hinted, ...rest];
}

export function resetMjhEpgCacheForTests(): void {
  xmlCache.clear();
  warmPending = null;
  resetParsedChannelCacheForTests();
}

let warmPending: Promise<void> | null = null;

/** Prefetch MJH XMLTV (mjh-all first) so EPG probes are fast after the first load. */
export function warmMjhEpgCaches(streamUrl = ""): Promise<void> {
  if (!warmPending) {
    const sources = orderedSourcesForStream(streamUrl);
    warmPending = Promise.all(
      sources.map(async (s) => {
        const xml = await fetchMjhXmltv(s.url);
        if (xml) parsedChannelsForXml(s.key, xml);
      }),
    ).then(() => undefined);
  }
  return warmPending;
}
