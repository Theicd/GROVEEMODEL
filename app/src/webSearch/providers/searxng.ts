import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

type SearxResult = {
  results?: Array<{ title?: string; url?: string; content?: string; engine?: string }>;
};

const getSearxngBaseUrl = (): string => {
  const env = import.meta.env.VITE_SEARXNG_URL as string | undefined;
  if (!env?.trim()) return "";
  const base = env.trim().replace(/\/$/, "");
  if (base.startsWith("/") && typeof window !== "undefined") {
    return `${window.location.origin}${base}`;
  }
  return base;
};

/** Open-domain fallback via self-hosted SearXNG JSON API (optional VITE_SEARXNG_URL). */
export const fetchSearxngSearch = async (query: string): Promise<SearchSourceResult> => {
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

  const url = `${base}/search?q=${encodeURIComponent(query.trim())}&format=json&language=he-IL`;
  try {
    const data = await fetchJson<SearxResult>(url, undefined, { timeoutMs: 12_000 });
    const hits = (data.results ?? []).slice(0, 6);
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
      const snippet = (r.content ?? "").replace(/\s+/g, " ").trim().slice(0, 120);
      return `${i + 1}. ${r.title ?? "ללא כותרת"} · ${r.url ?? ""}${snippet ? `\n   ${snippet}` : ""}`;
    });
    return {
      provider,
      label,
      ok: true,
      text: ["תוצאות חיפוש כללי (SearXNG):", ...lines].join("\n"),
      url: base,
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
