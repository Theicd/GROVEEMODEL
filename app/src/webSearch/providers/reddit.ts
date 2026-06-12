import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

const REDDIT_UA = "GROVEEMODEL/1.0 (browser chat; read-only)";

const extractSubreddit = (query: string): string | null => {
  const q = query.trim();
  const subMatch = q.match(/(?:^|[\s\-/])r\/([A-Za-z0-9_]+)/i);
  if (subMatch?.[1]) return subMatch[1];
  if (/worldnews|חדשות\s*עולם/i.test(q)) return "worldnews";
  if (/technology|טכנולוגיה/i.test(q)) return "technology";
  if (/israel|ישראל/i.test(q)) return "israel";
  return null;
};

export const fetchRedditSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "reddit" as const;
  const label = "Reddit";
  const sub = extractSubreddit(query);
  if (!sub) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "לא זוהה subreddit (למשל r/worldnews)",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const url = `https://www.reddit.com/r/${sub}/hot.json?limit=8&raw_json=1`;
    const data = await fetchJson<{
      data?: {
        children?: Array<{
          data?: { title?: string; score?: number; num_comments?: number; permalink?: string; url?: string };
        }>;
      };
    }>(url, {
      headers: { "User-Agent": REDDIT_UA },
    });

    const posts = data.data?.children ?? [];
    if (!posts.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `r/${sub}: אין פוסטים`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const text = posts
      .map((p) => {
        const d = p.data;
        if (!d?.title) return null;
        const link = d.permalink ? `https://www.reddit.com${d.permalink}` : d.url ?? "";
        return `- ${d.title} (↑${d.score ?? 0}, ${d.num_comments ?? 0} תגובות)\n  ${link}`;
      })
      .filter(Boolean)
      .join("\n");

    return {
      provider,
      label,
      ok: true,
      text: `r/${sub} — פוסטים חמים:\n${text}`,
      url: `https://www.reddit.com/r/${sub}/`,
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
