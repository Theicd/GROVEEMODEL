import { fetchText } from "../fetchJson";
import { extractNewsSite } from "../queryExtract";
import type { SearchSourceResult } from "../types";

const FEEDS: Record<string, { url: string; label: string }> = {
  bbc: {
    url: "https://feeds.bbci.co.uk/news/rss.xml",
    label: "BBC News",
  },
  cnn: {
    url: "http://rss.cnn.com/rss/edition.rss",
    label: "CNN",
  },
};

const parseRssTitles = (xml: string, limit = 5): string[] => {
  const titles: string[] = [];
  const re = /<item[\s\S]*?<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml)) && titles.length < limit) {
    const title = m[1].replace(/<[^>]+>/g, "").trim();
    if (title) titles.push(title);
  }
  return titles;
};

export const fetchNewsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "news-rss" as const;
  const label = "חדשות (RSS)";
  try {
    const site = extractNewsSite(query);
    if (!site || !FEEDS[site]) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהה מקור חדשות",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const feed = FEEDS[site];
    const xml = await fetchText(feed.url, { headers: { Accept: "application/rss+xml, application/xml, text/xml" } });
    const titles = parseRssTitles(xml, 5);
    if (!titles.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצאו כותרות",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `מקור: ${feed.label}`,
      `כותרות עדכניות (${new Date().toISOString().slice(0, 16)} UTC):`,
      ...titles.map((t, i) => `${i + 1}. ${t}`),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: feed.url,
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
