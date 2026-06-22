import {
  COMPANION_DEFAULT_LANG,
  COMPANION_DEFAULT_REGION,
  COMPANION_IMAGE_ENGINES,
  COMPANION_IMAGE_LIMIT,
  COMPANION_IMAGE_MODE,
  COMPANION_IMAGE_TIMEOUT_MS,
  COMPANION_PROBE_TIMEOUT_MS,
  COMPANION_WEB_ENGINES,
  COMPANION_WEB_LIMIT,
  COMPANION_WEB_MODE,
  COMPANION_WEB_TIMEOUT_MS,
} from "../../plugins/search-companion/companionConfig";
import {
  getSearchCompanionServiceUrl,
  resolveSearchCompanionFetchBase,
  setSearchCompanionUrl,
  usesDevOpenSerpProxy,
} from "../../plugins/search-companion/companionSettings";
import { isSearchCompanionReachable } from "../../plugins/search-companion/health";
import type { MediaSerpHit, SearchSourceResult, WebSerpHit } from "../types";
import { promoteCompanionWebHitsToMedia } from "./openserpWebMedia";

type OpenSerpResult = {
  id?: string;
  rank?: number;
  type?: string;
  title?: string;
  url?: string;
  display_url?: string;
  snippet?: string;
  domain?: string;
  engine?: string;
  image?: { url?: string; thumbnail?: string };
  source?: { page_url?: string; domain?: string };
};

type OpenSerpResponse = {
  results?: OpenSerpResult[];
  meta?: {
    engines_failed?: string[];
    engines_responded?: string[];
    took_ms?: number;
  };
};

export const getOpenSerpBaseUrl = (): string => getSearchCompanionServiceUrl();

export const setOpenSerpBaseUrl = setSearchCompanionUrl;

export const isOpenSerpConfigured = (): boolean =>
  !!getSearchCompanionServiceUrl() || isSearchCompanionReachable();

const normalizeOpenSerpWebHits = (rows: OpenSerpResult[]): WebSerpHit[] =>
  rows
    .filter((row) => row.url?.trim() && row.type !== "image")
    .map((row, index) => ({
      id: row.id || `openserp-${index}-${row.url?.slice(0, 64)}`,
      title: (row.title || "ללא כותרת").trim(),
      url: row.url!.trim(),
      snippet: (row.snippet || row.display_url || row.domain || "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 320),
      engine: row.engine,
    }));

const normalizeOpenSerpImageHits = (rows: OpenSerpResult[]): MediaSerpHit[] =>
  rows
    .filter((row) => row.image?.url || row.image?.thumbnail)
    .map((row, index) => {
      const pageUrl = row.source?.page_url || row.url || row.image?.url || "";
      const imageUrl = row.image?.url || row.image?.thumbnail || "";
      const thumb = row.image?.thumbnail || row.image?.url || imageUrl;
      return {
        id: row.id || `openserp-img-${index}-${thumb.slice(0, 48)}`,
        mediaType: "image" as const,
        title: (row.title || row.source?.domain || "תמונה").trim(),
        url: pageUrl || imageUrl,
        playUrl: imageUrl || thumb,
        thumbnail: thumb,
        snippet: row.source?.domain || row.engine || "",
        source: `OpenSERP${row.engine ? ` · ${row.engine}` : ""}`,
      };
    });

export type OpenSerpSearchOptions = {
  engines?: string;
  limit?: number;
  mode?: "any" | "fast" | "balanced";
  lang?: string;
  region?: string;
  extract?: 0 | 1 | 2;
  timeoutMs?: number;
  /** When false, skip parallel /mega/image (probe / fast paths). */
  includeImages?: boolean;
};

const formatOpenSerpFetchError = (err: unknown): string => {
  if (err instanceof DOMException && err.name === "AbortError") {
    return "OpenSERP לא הגיב בזמן — נסה שוב או הפעל מחדש את Grove Search (לפעמים Google/Bing איטיים).";
  }
  const msg = err instanceof Error ? err.message : String(err);
  if (/failed to fetch|networkerror|cors/i.test(msg)) {
    return "לא ניתן להגיע ל-OpenSERP מהדפדפן — ודא ש-Grove Search רץ על 127.0.0.1:7000 ובדוק ב-🧩 תוספים.";
  }
  if (/HTTP 502|offline/i.test(msg)) {
    return "OpenSERP לא זמין — הפעל «Grove Search» משולחן העבודה.";
  }
  if (/HTTP/i.test(msg)) {
    return `OpenSERP החזיר שגיאה (${msg.replace(/^HTTP\s+/i, "HTTP ")})`;
  }
  return msg.length > 120 ? `${msg.slice(0, 117)}…` : msg;
};

const fetchOpenSerpJson = async <T>(url: string, timeoutMs: number): Promise<T> => {
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return (await response.json()) as T;
  } finally {
    globalThis.clearTimeout(timer);
  }
};

const buildMegaUrl = (
  base: string,
  path: "search" | "image",
  query: string,
  options: OpenSerpSearchOptions,
  defaults: {
    engines: string;
    limit: number;
    mode: "any" | "fast" | "balanced";
    timeoutMs: number;
  },
): { url: string; timeoutMs: number } => {
  const params = new URLSearchParams({
    text: query.trim(),
    engines: options.engines || defaults.engines,
    limit: String(options.limit ?? defaults.limit),
    mode: options.mode || defaults.mode,
    lang: options.lang || COMPANION_DEFAULT_LANG,
    region: options.region || COMPANION_DEFAULT_REGION,
    dedupe: "true",
    merge: "true",
    format: "json",
  });
  if (options.extract) {
    params.set("extract", String(options.extract));
  }
  return {
    url: `${base}/mega/${path}?${params.toString()}`,
    timeoutMs: options.timeoutMs ?? defaults.timeoutMs,
  };
};

const fetchOpenSerpMegaWeb = async (
  query: string,
  options?: OpenSerpSearchOptions,
): Promise<{ data: OpenSerpResponse | null; error?: string }> => {
  const base = resolveSearchCompanionFetchBase();
  const { url, timeoutMs } = buildMegaUrl(base, "search", query, options ?? {}, {
    engines: COMPANION_WEB_ENGINES,
    limit: COMPANION_WEB_LIMIT,
    mode: COMPANION_WEB_MODE,
    timeoutMs: COMPANION_WEB_TIMEOUT_MS,
  });
  try {
    return { data: await fetchOpenSerpJson<OpenSerpResponse>(url, timeoutMs) };
  } catch (err) {
    return { data: null, error: formatOpenSerpFetchError(err) };
  }
};

const fetchOpenSerpMegaImages = async (
  query: string,
  options?: OpenSerpSearchOptions,
): Promise<{ mediaHits: MediaSerpHit[]; meta?: OpenSerpResponse["meta"] }> => {
  const base = resolveSearchCompanionFetchBase();
  const { url, timeoutMs } = buildMegaUrl(base, "image", query, options ?? {}, {
    engines: COMPANION_IMAGE_ENGINES,
    limit: COMPANION_IMAGE_LIMIT,
    mode: COMPANION_IMAGE_MODE,
    timeoutMs: COMPANION_IMAGE_TIMEOUT_MS,
  });
  try {
    const data = await fetchOpenSerpJson<OpenSerpResponse>(url, timeoutMs);
    return {
      mediaHits: normalizeOpenSerpImageHits((data.results || []).slice(0, COMPANION_IMAGE_LIMIT)),
      meta: data.meta,
    };
  } catch {
    return { mediaHits: [] };
  }
};

const formatWebLines = (hits: WebSerpHit[], enginesFailed?: string[]): string => {
  const lines = hits.map((hit, index) => {
    const snippet = hit.snippet ? `\n   ${hit.snippet.slice(0, 160)}` : "";
    const engine = hit.engine ? ` · ${hit.engine}` : "";
    return `${index + 1}. ${hit.title}${engine}\n   ${hit.url}${snippet}`;
  });
  const failed = enginesFailed?.length ? `\nמנועים שנכשלו: ${enginesFailed.join(", ")}` : "";
  return ["תוצאות חיפוש כללי (OpenSERP):", ...lines, failed].filter(Boolean).join("\n");
};

/** Web + image megasearch via local OpenSERP companion. */
export const fetchOpenSerpSearch = async (
  query: string,
  options?: OpenSerpSearchOptions,
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "openserp" as const;
  const label = "OpenSERP (web)";
  const includeImages = options?.includeImages !== false;

  const [webOut, imageOut] = await Promise.all([
    fetchOpenSerpMegaWeb(query, options),
    includeImages ? fetchOpenSerpMegaImages(query, options) : Promise.resolve({ mediaHits: [] }),
  ]);

  const webData = webOut.data;
  const rawWebRows = (webData?.results || []).slice(0, COMPANION_WEB_LIMIT);
  const { webHits: promotedWeb, mediaHits: videoMedia } = await promoteCompanionWebHitsToMedia(
    normalizeOpenSerpWebHits(rawWebRows),
  );
  const imageMedia = imageOut.mediaHits ?? [];
  const mediaHits = [...videoMedia, ...imageMedia];

  const enginesFailed = [
    ...(webData?.meta?.engines_failed ?? []),
    ...(imageOut.meta?.engines_failed ?? []),
  ].filter((v, i, arr) => arr.indexOf(v) === i);

  if (!promotedWeb.length && !mediaHits.length) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      url: getSearchCompanionServiceUrl(),
      error:
        webOut.error ||
        (enginesFailed.length
          ? `OpenSERP — מנועים נכשלו: ${enginesFailed.join(", ")}`
          : "OpenSERP לא החזיר תוצאות."),
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const imageNote =
    imageMedia.length > 0 ? `\nתמונות (OpenSERP): ${imageMedia.length} מתוך Google/Bing/Duck` : "";
  const videoNote =
    videoMedia.length > 0 ? `\nוידאו מקושר: ${videoMedia.length} (Archive/Vimeo/PeerTube…)` : "";

  return {
    provider,
    label,
    ok: true,
    text: formatWebLines(promotedWeb, enginesFailed) + imageNote + videoNote,
    url: getSearchCompanionServiceUrl(),
    webHits: promotedWeb.length ? promotedWeb : undefined,
    mediaHits: mediaHits.length ? mediaHits : undefined,
    latencyMs: Math.round(performance.now() - started),
  };
};

export const probeOpenSerpSearch = async (
  query = "webgpu browser",
): Promise<{ ok: boolean; messageHe: string; hitCount?: number }> => {
  const result = await fetchOpenSerpSearch(query, {
    limit: 3,
    mode: "fast",
    engines: "bing,duckduckgo,google",
    includeImages: false,
    timeoutMs: COMPANION_PROBE_TIMEOUT_MS,
  });
  if (result.ok && (result.webHits?.length || result.mediaHits?.length)) {
    const n = (result.webHits?.length ?? 0) + (result.mediaHits?.length ?? 0);
    const via = usesDevOpenSerpProxy() ? " (proxy מקומי)" : "";
    return {
      ok: true,
      messageHe: `חיפוש עובד${via} — ${n} תוצאות (${result.latencyMs}ms)`,
      hitCount: n,
    };
  }
  return {
    ok: false,
    messageHe: result.error || "חיפוש נכשל",
  };
};

export const probeOpenSerpConnection = async (): Promise<{ ok: boolean; messageHe: string }> => {
  const { probeSearchCompanionHealth } = await import("../../plugins/search-companion/health");
  const health = await probeSearchCompanionHealth();
  return {
    ok: health.status === "online" || health.status === "degraded",
    messageHe: health.messageHe,
  };
};
