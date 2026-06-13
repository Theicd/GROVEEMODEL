import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

type HnItem = { title?: string; url?: string; score?: number; id?: number };

export const fetchHackerNewsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "hacker-news" as const;
  const label = "Hacker News";
  const wantTop =
    /(?:hacker\s*news|hn\b|ycombinator)/i.test(query) &&
    /(?:פופולרי|popular|top|best|מוביל|הכי|כרגע|now|כותרת|headline|פוסט)/i.test(query);

  try {
    const ids = await fetchJson<number[]>("https://hacker-news.firebaseio.com/v0/topstories.json");
    const top = ids.slice(0, wantTop ? 5 : 12);
    const items = await Promise.all(
      top.map((id) =>
        fetchJson<HnItem>(`https://hacker-news.firebaseio.com/v0/item/${id}.json`, undefined, {
          timeoutMs: 8000,
        }),
      ),
    );

    const sorted = [...items].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    if (wantTop && sorted[0]) {
      const topStory = sorted[0];
      const text = [
        "הפוסט המוביל כרגע ב-Hacker News:",
        `1. ${topStory.title ?? "—"} (★${topStory.score ?? 0}) ${topStory.url ?? ""}`.trim(),
        sorted[1]
          ? `2. ${sorted[1].title ?? "—"} (★${sorted[1].score ?? 0})`
          : "",
      ]
        .filter(Boolean)
        .join("\n");
      return {
        provider,
        label,
        ok: true,
        text,
        url: topStory.url ?? "https://news.ycombinator.com",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const qLower = query.toLowerCase();
    const tokens = qLower.split(/\s+/).filter((t) => t.length > 3 && !/hacker|news|ycombinator|פופולרי|popular/.test(t));
    const filtered =
      tokens.length > 0
        ? items.filter((it) => {
            const title = (it.title ?? "").toLowerCase();
            return tokens.some((t) => title.includes(t));
          })
        : items;

    const picked = (filtered.length ? filtered : items).slice(0, 6);
    if (!picked.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין כותרות",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const text = picked
      .map((it, i) => `${i + 1}. ${it.title ?? "—"} (★${it.score ?? 0}) ${it.url ?? ""}`.trim())
      .join("\n");

    return {
      provider,
      label,
      ok: true,
      text,
      url: "https://news.ycombinator.com",
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
