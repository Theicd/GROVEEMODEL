import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

export const fetchHackerNewsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "hackernews" as const;
  const label = "Hacker News";
  const q = query.trim().replace(/hacker\s*news|hn\b|ycombinator/gi, " ").trim() || query.trim();

  try {
    const url = `https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(q)}&tags=story&hitsPerPage=8`;
    const data = await fetchJson<{
      hits?: Array<{ title?: string; url?: string; points?: number; num_comments?: number; objectID?: string }>;
    }>(url);

    const hits = data.hits ?? [];
    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין תוצאות",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const text = hits
      .map((h) => {
        const link = h.url || (h.objectID ? `https://news.ycombinator.com/item?id=${h.objectID}` : "");
        return `- ${h.title ?? "(ללא כותרת)"} (↑${h.points ?? 0}, ${h.num_comments ?? 0} תגובות)\n  ${link}`;
      })
      .join("\n");

    return {
      provider,
      label,
      ok: true,
      text: `HN — "${q}":\n${text}`,
      url: `https://hn.algolia.com/?q=${encodeURIComponent(q)}`,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
