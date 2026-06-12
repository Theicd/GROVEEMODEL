import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

/** Public SearXNG instances — JSON API via CORS relay on static hosts. */
const SEARX_INSTANCES = [
  "https://searx.be",
  "https://search.bus-hit.me",
  "https://searx.tiekoetter.com",
];

type SearxResult = {
  results?: Array<{ title?: string; url?: string; content?: string; engine?: string }>;
};

export const fetchSearxSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "searxng" as const;
  const label = "SearXNG (Web)";
  const q = query.trim();
  if (!q) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "שאילתה ריקה",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  let lastErr = "אין תוצאות";
  for (const base of SEARX_INSTANCES) {
    try {
      const url = `${base}/search?q=${encodeURIComponent(q)}&format=json&language=he-IL`;
      const data = await fetchJson<SearxResult>(url);
      const hits = (data.results ?? []).slice(0, 6);
      if (!hits.length) continue;

      const text = hits
        .map((h, i) => {
          const snippet = (h.content ?? "").replace(/\s+/g, " ").trim().slice(0, 220);
          return `${i + 1}. ${h.title ?? "(ללא כותרת)"}\n   ${snippet}${snippet ? "…" : ""}\n   ${h.url ?? ""}${h.engine ? ` [${h.engine}]` : ""}`;
        })
        .join("\n\n");

      return {
        provider,
        label,
        ok: true,
        text: `שאילתה: ${q}\n${text}`,
        url: `${base}/search?q=${encodeURIComponent(q)}`,
        latencyMs: Math.round(performance.now() - started),
      };
    } catch (err) {
      lastErr = err instanceof Error ? err.message : "שגיאה";
    }
  }

  return {
    provider,
    label,
    ok: false,
    text: "",
    error: lastErr,
    latencyMs: Math.round(performance.now() - started),
  };
};
