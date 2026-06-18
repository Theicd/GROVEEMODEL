import { fetchJson } from "../fetchJson";
import type { SearchSourceResult, WebSerpHit } from "../types";
import { promoteArchiveWebHitsToMedia } from "./archiveOrgVideo";

type SearxResult = {
  results?: Array<{ title?: string; url?: string; content?: string; engine?: string }>;
};

export const getSearxngBaseUrl = (): string => {
  const env = import.meta.env.VITE_SEARXNG_URL as string | undefined;
  if (!env?.trim()) return "";
  const base = env.trim().replace(/\/$/, "");
  if (base.startsWith("/") && typeof window !== "undefined") {
    return `${window.location.origin}${base}`;
  }
  return base;
};

const toWebHits = (
  rows: Array<{ title?: string; url?: string; content?: string; engine?: string }>,
): WebSerpHit[] =>
  rows
    .filter((r) => r.url?.trim())
    .map((r, i) => ({
      id: `searxng-${i}-${(r.url ?? "").slice(0, 48)}`,
      title: (r.title ?? "ללא כותרת").trim(),
      url: r.url!.trim(),
      snippet: (r.content ?? "").replace(/\s+/g, " ").trim().slice(0, 280),
      engine: r.engine,
    }));

/** Open-domain fallback via self-hosted SearXNG JSON API (optional VITE_SEARXNG_URL). */
export const fetchSearxngSearch = async (
  query: string,
  category = "general",
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "searxng" as const;
  const label = "SearXNG (web)";
  const base = getSearxngBaseUrl();

  if (!base) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "SearXNG לא מוגדר — הגדר VITE_SEARXNG_URL",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const params = new URLSearchParams({
    q: query.trim(),
    format: "json",
    language: "he-IL",
    categories: category,
  });
  const url = `${base}/search?${params.toString()}`;
  try {
    const data = await fetchJson<SearxResult>(url, undefined, { timeoutMs: 12_000 });
    const rawHits = toWebHits((data.results ?? []).slice(0, 15));
    const { webHits: hits, mediaHits } = await promoteArchiveWebHitsToMedia(rawHits);
    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין תוצאות SearXNG",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const lines = hits.map((r, i) => {
      const snippet = r.snippet.slice(0, 120);
      return `${i + 1}. ${r.title} · ${r.url}${snippet ? `\n   ${snippet}` : ""}`;
    });
    return {
      provider,
      label,
      ok: true,
      text: ["תוצאות חיפוש כללי (SearXNG):", ...lines].join("\n"),
      url: base,
      webHits: hits,
      mediaHits: mediaHits.length ? mediaHits : undefined,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : String(err),
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

export const isSearxngConfigured = (): boolean => !!getSearxngBaseUrl();
