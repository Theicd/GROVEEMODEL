import { fetchText } from "../fetchJson";
import { extractNewsSite, isWorldHeadlineQuery } from "../queryExtract";
import type { SearchSourceResult } from "../types";

type NewsFeed = { url: string; label: string; tag: string };

const FEEDS: Record<string, NewsFeed> = {
  bbc: {
    url: "https://feeds.bbci.co.uk/news/rss.xml",
    label: "BBC News",
    tag: "BBC",
  },
  cnn: {
    url: "http://rss.cnn.com/rss/edition.rss",
    label: "CNN",
    tag: "CNN",
  },
  reuters: {
    url: "https://feeds.reuters.com/reuters/worldNews",
    label: "Reuters World",
    tag: "Reuters",
  },
  guardian: {
    url: "https://www.theguardian.com/world/rss",
    label: "The Guardian World",
    tag: "Guardian",
  },
  ynet: {
    url: "https://www.ynet.co.il/Integration/StoryRss2.xml",
    label: "ynet",
    tag: "ynet",
  },
};

const WORLD_FEED_KEYS = ["bbc", "cnn", "reuters", "guardian"] as const;

export const parseRssTitles = (xml: string, limit = 5): string[] => {
  const titles: string[] = [];
  const re = /<item[\s\S]*?<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml)) && titles.length < limit) {
    const title = m[1].replace(/<[^>]+>/g, "").trim();
    if (title) titles.push(title);
  }
  return titles;
};

const fetchFeedTitles = async (feed: NewsFeed, limit: number): Promise<string[]> => {
  const xml = await fetchText(
    feed.url,
    { headers: { Accept: "application/rss+xml, application/xml, text/xml" } },
    { timeoutMs: 12_000 },
  );
  return parseRssTitles(xml, limit);
};

const fetchWorldHeadlines = async (): Promise<Array<{ feed: NewsFeed; titles: string[] }>> => {
  const results = await Promise.allSettled(
    WORLD_FEED_KEYS.map(async (key) => {
      const feed = FEEDS[key];
      const titles = await fetchFeedTitles(feed, 2);
      return { feed, titles };
    }),
  );

  return results
    .filter((r): r is PromiseFulfilledResult<{ feed: NewsFeed; titles: string[] }> => r.status === "fulfilled")
    .map((r) => r.value)
    .filter((row) => row.titles.length > 0);
};

export const fetchNewsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "news-rss" as const;
  const label = "חדשות (RSS)";

  try {
    if (isWorldHeadlineQuery(query)) {
      const rows = await fetchWorldHeadlines();
      if (!rows.length) {
        return {
          provider,
          label,
          ok: false,
          text: "",
          error: "לא נמצאו כותרות ממקורות RSS",
          latencyMs: Math.round(performance.now() - started),
        };
      }

      const top = rows[0];
      const topTitle = top.titles[0];
      const sourceTags = rows.map((r) => r.feed.tag).join(" · ");
      const lines = [
        `ANSWER (headline): [${top.feed.tag}] ${topTitle}`,
        `מקורות RSS בינלאומיים (${sourceTags}):`,
        `עודכן: ${new Date().toISOString().replace("T", " ").slice(0, 19)} UTC`,
        ...rows.flatMap(({ feed, titles }) =>
          titles.map((t, i) => `[${feed.tag}] ${i + 1}. ${t}`),
        ),
      ];

      return {
        provider,
        label: "חדשות (RSS — BBC · CNN · Reuters · Guardian)",
        ok: true,
        text: lines.join("\n"),
        url: FEEDS.bbc.url,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const site = extractNewsSite(query) ?? "bbc";
    if (!FEEDS[site]) {
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
    const titles = await fetchFeedTitles(feed, 5);
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
      `ANSWER (headline): [${feed.tag}] ${titles[0]}`,
      `מקור: ${feed.label}`,
      `כותרות עדכניות (${new Date().toISOString().slice(0, 16)} UTC):`,
      ...titles.map((t, i) => `[${feed.tag}] ${i + 1}. ${t}`),
    ];

    return {
      provider,
      label: `חדשות (${feed.label})`,
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
