import type { SearchSourceResult } from "../webSearch/types";
import { resolveNetworkReachability } from "../networkReachability";
import { isTopicsOverviewQuery } from "./headlineIntent";
import { queryHasHebrew } from "./hebrewSearchTerms";
import {
  isBroadNewsOverviewQuery,
  isSensorNewsQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import { isNewsQuery } from "../webSearch/intents";
import { buildRecentHeadlineHits } from "./recentHeadlineHits";
import { hitsToDisplayCards } from "./searchAdapter";
import { fetchTopicsBundle } from "./topicsAdapter";
import { startGroveeNewsBoot } from "./engineBoot";
import type { GroveeNewsCard } from "./types";
import { searchNews } from "./engine/engine/pipeline";
import { pollRssForLiveSearch } from "./liveSearchPoll";
import { getEngineLibraryStats } from "./engine/engine/engineStats";
import { getSearchIndexSize } from "./engine/search/flexIndex";
import { isTopHitRelevant } from "./engine/search/relevance";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import { getRssHeadlineCount } from "./engine/storage/db";
import type { SearchHit } from "./engine/types";

const PROVIDER = "grovee-news" as const;

export type GroveeNewsSearchOptions = {
  onRefresh?: (result: SearchSourceResult) => void;
};

function formatHeadlinesForBrief(cards: { title: string; source: string }[]): string {
  const lines = cards.slice(0, 10).map((c, i) => `[${c.source}] ${i + 1}. ${c.title}`);
  if (!lines.length) return "";
  const lead = cards[0];
  return [
    `ANSWER (headline): [${lead.source}] ${lead.title}`,
    "מקור: GROVEE NEWS (סריקת RSS)",
    ...lines,
  ].join("\n");
}

function mergeSearchHits(primary: SearchHit[], extra: SearchHit[]): SearchHit[] {
  const byUrl = new Map<string, SearchHit>();
  for (const hit of [...primary, ...extra]) {
    const key = hit.article.url;
    const prev = byUrl.get(key);
    if (!prev || hit.score > prev.score) byUrl.set(key, hit);
  }
  return [...byUrl.values()].sort((a, b) => b.score - a.score);
}

async function searchNewsWithFallback(query: string): Promise<SearchHit[]> {
  const engineQuery = normalizeNewsEngineQuery(query);
  const specific = isSpecificNewsTopicQuery(query);
  const heUi = getUserNewsProfile().uiLanguage === "he";
  const searchQ = engineQuery || query;

  if (specific && !searchQ.trim()) return [];

  let hits = await searchNews(searchQ);

  if (queryHasHebrew(query) && engineQuery && engineQuery !== query) {
    const heHits = await searchNews(query);
    hits = mergeSearchHits(hits, heHits);
  }

  if (engineQuery && specific) {
    const relevanceQuery = isSensorNewsQuery(query) ? engineQuery : query;
    hits = hits.filter((h) => isTopHitRelevant(h.article, relevanceQuery));
  }

  if (!hits.length && engineQuery && engineQuery !== query && !specific) {
    hits = await searchNews(query);
  }

  if (!hits.length && (!specific || isBroadNewsOverviewQuery(engineQuery))) {
    hits = await buildRecentHeadlineHits(query);
  }

  if (!hits.length && (specific || isNewsQuery(query) || isBroadNewsOverviewQuery(engineQuery))) {
    hits = await buildRecentHeadlineHits(query);
  }

  if (!hits.length && !specific) {
    hits = await buildRecentHeadlineHits(query, 16);
  }

  if (!hits.length) {
    const headlineCount = await getRssHeadlineCount();
    if (headlineCount > 0) {
      hits = await buildRecentHeadlineHits(query, 12);
    }
  }

  if (heUi && hits.length) {
    hits = [...hits].sort((a, b) => {
      const aIl = (a.article.sourceKey ?? "").startsWith("il_") ? 1 : 0;
      const bIl = (b.article.sourceKey ?? "").startsWith("il_") ? 1 : 0;
      if (aIl !== bIl) return bIl - aIl;
      return b.score - a.score;
    });
  }

  return hits;
}

function buildNewsSearchResult(
  query: string,
  cards: GroveeNewsCard[],
  stats: { rssHeadlines: number },
  started: number,
  scanNote: string,
  label = "חדשות (GROVEE NEWS)",
): SearchSourceResult {
  const text =
    cards.length > 0
      ? formatHeadlinesForBrief(cards)
      : `סריקת RSS · ${stats.rssHeadlines} כותרות במאגר — לא נמצאו תוצאות ל«${query.slice(0, 80)}»`;

  return {
    provider: PROVIDER,
    label,
    ok: cards.length > 0,
    text,
    url: cards[0]?.url,
    newsCards: cards,
    newsScanNote: scanNote,
    error: cards.length ? undefined : "לא נמצאו ידיעות תואמות לאחר סריקת RSS",
    latencyMs: Math.round(performance.now() - started),
  };
}

/** Search or Topics → SearchSourceResult + side panel payload. */
export async function fetchGroveeNewsSearch(
  query: string,
  _options?: GroveeNewsSearchOptions,
): Promise<SearchSourceResult> {
  const started = performance.now();
  const label = "חדשות (GROVEE NEWS)";

  try {
    const reachability = await resolveNetworkReachability();
    if (reachability === "offline") {
      return {
        provider: PROVIDER,
        label,
        ok: false,
        text: "",
        newsScanNote: "אין חיבור לאינטרנט — חיפוש חדשות דורש רשת",
        error: "אין חיבור לאינטרנט — חיפוש חדשות דורש רשת",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    await startGroveeNewsBoot();

    const poll = await pollRssForLiveSearch(query);
    const stats = await getEngineLibraryStats(getSearchIndexSize());
    const scanNote =
      poll.feedsOk > 0
        ? `סריקת RSS · ${poll.feedsOk} מקורות · ${stats.rssHeadlines.toLocaleString("he-IL")} כותרות במאגר`
        : "סריקת RSS — לא הצלחנו לגשת למקורות (בדוק חיבור)";

    if (isTopicsOverviewQuery(query)) {
      const bundle = await fetchTopicsBundle();
      const cards: GroveeNewsCard[] = bundle.cards.slice(0, 24);
      const text =
        cards.length > 0
          ? formatHeadlinesForBrief(cards)
          : `סריקת RSS · Topics עדיין מתמלא (${stats.rssHeadlines} כותרות במאגר)`;

      return {
        provider: PROVIDER,
        label: `חדשות (Topics · ${bundle.stats.lanesWithHits || cards.length} נושאים)`,
        ok: cards.length > 0,
        text,
        newsCards: cards,
        newsScanNote: scanNote,
        error: cards.length ? undefined : "אין עדיין נושאים מוכנים",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const hits = await searchNewsWithFallback(query);
    const cards = await hitsToDisplayCards(hits);
    return buildNewsSearchResult(query, cards, stats, started, scanNote, label);
  } catch (err) {
    return {
      provider: PROVIDER,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאת מנוע חדשות",
      latencyMs: Math.round(performance.now() - started),
    };
  }
}
