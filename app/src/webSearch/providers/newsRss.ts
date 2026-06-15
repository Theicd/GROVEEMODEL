import { fetchText } from "../fetchJson";
import { defaultFetchTimeoutMs } from "../proxyFetch";
import {
  extractNewsSite,
  isGeneralNewsDigestQuery,
  isIsraelNewsQuery,
  isWorldHeadlineQuery,
} from "../queryExtract";
import type { SearchSourceResult } from "../types";
import { agentDebugLog } from "../../debugAgentLog";
import {
  feedKeyFromSourceLabel,
  AI_NEWS_KEYS,
  ENTERTAINMENT_KEYS,
  getFeedByKey,
  ISRAEL_BUSINESS_KEYS,
  ISRAEL_DIGEST_KEYS,
  ISRAEL_NEWS_KEYS,
  ISRAEL_SPORT_KEYS,
  ISRAEL_TECH_KEYS,
  NEWS_FEEDS,
  SPACE_SCIENCE_KEYS,
  sortFeedKeysByPriority,
  TREND_KEYS,
  WORLD_BUSINESS_KEYS,
  WORLD_DIGEST_KEYS,
  WORLD_HEADLINE_KEYS,
  WORLD_TECH_KEYS,
  type NewsFeedDef,
} from "./newsFeeds";

export { NEWS_FEEDS, WORLD_HEADLINE_KEYS, ISRAEL_NEWS_KEYS } from "./newsFeeds";

export const parseRssTitles = (xml: string, limit = 5): string[] => {
  const titles: string[] = [];
  const decode = (raw: string) =>
    raw
      .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/gi, "$1")
      .replace(/<[^>]+>/g, "")
      .trim();

  const rssRe = /<item[\s\S]*?<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/gi;
  let m: RegExpExecArray | null;
  while ((m = rssRe.exec(xml)) && titles.length < limit) {
    const title = decode(m[1]);
    if (title) titles.push(title);
  }
  if (titles.length) return titles;

  const atomRe = /<entry[\s\S]*?<title[^>]*>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/gi;
  while ((m = atomRe.exec(xml)) && titles.length < limit) {
    const title = decode(m[1]);
    if (title) titles.push(title);
  }
  return titles;
};

const fetchFeedTitles = async (feed: NewsFeedDef, limit: number): Promise<string[]> => {
  const urls = [feed.url, ...(feed.fallbackUrls ?? [])];
  let lastErr: unknown;
  for (const url of urls) {
    try {
      const xml = await fetchText(
        url,
        { headers: { Accept: "application/rss+xml, application/xml, text/xml, application/atom+xml" } },
        { timeoutMs: typeof window !== "undefined" ? defaultFetchTimeoutMs() : 14_000 },
      );
      const titles = parseRssTitles(xml, limit);
      if (titles.length) return titles;
      lastErr = new Error("לא נמצאו כותרות");
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("שגיאה");
};

/** Which RSS feeds to hit — region-aware; world queries exclude Israeli feeds. */
export const selectNewsFeedKeys = (query: string): string[] => {
  const site = extractNewsSite(query);
  const isIsrael = isIsraelNewsQuery(query);
  const topicKeys = selectTopicFeedKeys(query, isIsrael);
  const flags = {
    site,
    israel: isIsrael,
    world: isWorldHeadlineQuery(query),
    digest: isGeneralNewsDigestQuery(query),
    topicKeys,
  };
  if (site && NEWS_FEEDS[site]) {
    const keys = [site];
    // #region agent log
    agentDebugLog("H2", "newsRss.ts:selectNewsFeedKeys", "news feed selected by explicit site", { queryPreview: query.slice(0, 120), flags, keys });
    // #endregion
    return keys;
  }

  if (topicKeys.length) {
    const keys = sortFeedKeysByPriority(topicKeys);
    // #region agent log
    agentDebugLog("H2", "newsRss.ts:selectNewsFeedKeys", "news feed selected for topic query", { queryPreview: query.slice(0, 120), flags, keys });
    // #endregion
    return keys;
  }

  if (isIsrael) {
    const keys = [...ISRAEL_NEWS_KEYS];
    // #region agent log
    agentDebugLog("H2", "newsRss.ts:selectNewsFeedKeys", "news feed selected for Israel query", { queryPreview: query.slice(0, 120), flags, keys });
    // #endregion
    return keys;
  }

  if (isWorldHeadlineQuery(query)) {
    const keys = [...WORLD_HEADLINE_KEYS];
    // #region agent log
    agentDebugLog("H2", "newsRss.ts:selectNewsFeedKeys", "news feed selected for world headline query", { queryPreview: query.slice(0, 120), flags, keys });
    // #endregion
    return keys;
  }

  if (isGeneralNewsDigestQuery(query)) {
    const keys = sortFeedKeysByPriority([...WORLD_DIGEST_KEYS, ...ISRAEL_DIGEST_KEYS]);
    // #region agent log
    agentDebugLog("H2", "newsRss.ts:selectNewsFeedKeys", "news feed selected for general digest query", { queryPreview: query.slice(0, 120), flags, keys });
    // #endregion
    return keys;
  }

  // Generic news — international first, one Israeli for local context
  const keys = sortFeedKeysByPriority([...WORLD_DIGEST_KEYS, "ynet"]);
  // #region agent log
  agentDebugLog("H2", "newsRss.ts:selectNewsFeedKeys", "news feed selected by generic fallback", { queryPreview: query.slice(0, 120), flags, keys });
  // #endregion
  return keys;
};

const selectTopicFeedKeys = (query: string, isIsrael: boolean): string[] => {
  const q = query.toLowerCase();
  const keys: string[] = [];
  if (/ai|בינה\s+מלאכותית|למידת\s+מכונה|llm|openai|anthropic|deepmind|hugging\s*face/i.test(query)) {
    keys.push(...AI_NEWS_KEYS);
  }
  if (/טכנולוג|tech|סטארט|startup|גאדג|סייבר|cyber|מחשבים|software|תוכנה/i.test(query)) {
    keys.push(...(isIsrael ? ISRAEL_TECH_KEYS : []), ...WORLD_TECH_KEYS);
  }
  if (/חלל|space|nasa|esa|מדע|science|אסטרונומ/i.test(query)) {
    keys.push(...SPACE_SCIENCE_KEYS);
  }
  if (/כלכל|עסק|business|markets?|בורסה|מניות|finance|financial|כסף|שוק/i.test(query)) {
    keys.push(...(isIsrael ? ISRAEL_BUSINESS_KEYS : []), ...WORLD_BUSINESS_KEYS);
  }
  if (/ספורט|sport|כדורגל|כדורסל/i.test(query)) {
    keys.push(...ISRAEL_SPORT_KEYS);
  }
  if (/בידור|תרבות|משחקים|גיימינג|gaming|games?|movies?|film|music|סרטים/i.test(query)) {
    keys.push(...ENTERTAINMENT_KEYS);
  }
  if (/טרנד|trending|popular|קהיל|reddit|product\s*hunt|github\s+trending|האקרים|hacker\s*news/i.test(q)) {
    keys.push(...TREND_KEYS);
  }
  return [...new Set(keys)];
};

/** Single feed — one SearchSourceResult (shown separately in UI). */
export const fetchNewsFeedByKey = async (key: string, limit = 3): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "news-rss" as const;
  const feed = getFeedByKey(key);
  if (!feed) {
    return {
      provider,
      label: "חדשות (RSS)",
      ok: false,
      text: "",
      error: `feed ${key} unknown`,
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const titles = await fetchFeedTitles(feed, limit);
    if (!titles.length) {
      return {
        provider,
        label: `חדשות (${feed.label})`,
        ok: false,
        text: "",
        error: "לא נמצאו כותרות",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `ANSWER (headline): [${feed.tag}] ${titles[0]}`,
      `מקור: ${feed.label}`,
      `אזור: ${feed.region}`,
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
      label: `חדשות (${feed.label})`,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

const pickLeadSource = (
  ok: SearchSourceResult[],
  preferWorld: boolean,
): SearchSourceResult => {
  const ranked = sortFeedKeysByPriority(
    ok.map((s) => feedKeyFromSourceLabel(s.label) ?? "").filter(Boolean),
  );
  const preferredKey = preferWorld
    ? ranked.find((k) => NEWS_FEEDS[k]?.region !== "israel")
    : ranked.find((k) => NEWS_FEEDS[k]?.region === "israel");
  if (preferredKey) {
    const match = ok.find((s) => feedKeyFromSourceLabel(s.label) === preferredKey);
    if (match) return match;
  }
  return ok[0]!;
};

/** Legacy aggregate — merges per-feed results for multi-feed queries. */
export const fetchNewsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "news-rss" as const;
  const label = "חדשות (RSS)";

  try {
    const keys = selectNewsFeedKeys(query);
    if (keys.length > 1) {
      const settled = await Promise.all(keys.map((k) => fetchNewsFeedByKey(k, 2)));
      const ok = settled.filter((s) => s.ok && s.text.trim());
      if (!ok.length) {
        return {
          provider,
          label,
          ok: false,
          text: "",
          error: "לא נמצאו כותרות ממקורות RSS",
          latencyMs: Math.round(performance.now() - started),
        };
      }
      const preferWorld = isWorldHeadlineQuery(query) || (!isIsraelNewsQuery(query) && !extractNewsSite(query));
      const top = pickLeadSource(ok, preferWorld);
      const sourceTags = ok.map((s) => s.label.replace("חדשות (", "").replace(")", "")).join(" · ");
      const lines = [
        top.text.match(/ANSWER \(headline\):.+/m)?.[0] ?? "",
        `מקורות RSS (${sourceTags}):`,
        ...ok.flatMap((s) =>
          s.text.split("\n").filter((l) => /^\[[^\]]+\]\s*\d+\./.test(l.trim())),
        ),
      ].filter(Boolean);
      return {
        provider,
        label: `חדשות (RSS — ${sourceTags})`,
        ok: true,
        text: lines.join("\n"),
        url: getFeedByKey("bbc")?.url,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    return fetchNewsFeedByKey(keys[0]!, 5);
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
