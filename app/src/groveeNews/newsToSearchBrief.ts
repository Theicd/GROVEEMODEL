import type { SearchSourceResult } from "../webSearch/types";
import { isTopicsOverviewQuery } from "./headlineIntent";
import { queryHasHebrew } from "./hebrewSearchTerms";
import {
  isBroadNewsOverviewQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import { buildRecentHeadlineHits } from "./recentHeadlineHits";
import { hitsToDisplayCards } from "./searchAdapter";
import { fetchTopicsBundle } from "./topicsAdapter";
import { startGroveeNewsBoot } from "./engineBoot";
import type { GroveeNewsCard } from "./types";
import { searchNews } from "./engine/engine/pipeline";
import { priorityPollHebrewFeeds } from "./hebrewFeedPoll";
import { getEngineLibraryStats } from "./engine/engine/engineStats";
import { getSearchIndexSize } from "./engine/search/flexIndex";
import { isTopHitRelevant } from "./engine/search/relevance";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import type { SearchHit } from "./engine/types";

const PROVIDER = "grovee-news" as const;

export type GroveeNewsSearchOptions = {
  /** Called when a background Hebrew RSS scan finds more relevant headlines. */
  onRefresh?: (result: SearchSourceResult) => void;
};

function formatHeadlinesForBrief(cards: { title: string; source: string }[]): string {
  const lines = cards.slice(0, 10).map((c, i) => `[${c.source}] ${i + 1}. ${c.title}`);
  if (!lines.length) return "";
  const lead = cards[0];
  return [
    `ANSWER (headline): [${lead.source}] ${lead.title}`,
    "מקור: GROVEE NEWS (מנוע מקומי)",
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
    hits = hits.filter((h) => isTopHitRelevant(h.article, query));
  }

  if (!hits.length && engineQuery && engineQuery !== query && !specific) {
    hits = await searchNews(query);
  }

  if (!hits.length && (!specific || isBroadNewsOverviewQuery(engineQuery))) {
    hits = await buildRecentHeadlineHits(query);
  }

  if (!hits.length && specific) {
    hits = await buildRecentHeadlineHits(query);
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
  label = "חדשות (GROVEE NEWS)",
): SearchSourceResult {
  const text =
    cards.length > 0
      ? formatHeadlinesForBrief(cards)
      : `מאגר: ${stats.rssHeadlines} כותרות RSS, אינדקס ${getSearchIndexSize()} — לא נמצאו תוצאות ל«${query.slice(0, 80)}»`;

  return {
    provider: PROVIDER,
    label,
    ok: cards.length > 0,
    text,
    url: cards[0]?.url,
    newsCards: cards,
    error: cards.length ? undefined : "לא נמצאו ידיעות תואמות במאגר המקומי",
    latencyMs: Math.round(performance.now() - started),
  };
}

async function continueHebrewNewsScan(
  query: string,
  baselineCount: number,
  onRefresh: (result: SearchSourceResult) => void,
): Promise<void> {
  const added = await priorityPollHebrewFeeds({ timeoutMs: 20_000 });
  if (added <= 0 && baselineCount > 0) return;

  const hits = await searchNewsWithFallback(query);
  if (hits.length <= baselineCount) return;

  const cards = await hitsToDisplayCards(hits);
  const stats = await getEngineLibraryStats(getSearchIndexSize());
  onRefresh(buildNewsSearchResult(query, cards, stats, performance.now()));
}

/** Search or Topics → SearchSourceResult + side panel payload. */
export async function fetchGroveeNewsSearch(
  query: string,
  options?: GroveeNewsSearchOptions,
): Promise<SearchSourceResult> {
  const started = performance.now();
  const label = "חדשות (GROVEE NEWS)";

  try {
    await startGroveeNewsBoot();

    const heUi = getUserNewsProfile().uiLanguage === "he";
    if (heUi) {
      await priorityPollHebrewFeeds({ timeoutMs: 12_000 });
    }

    const stats = await getEngineLibraryStats(getSearchIndexSize());

    if (isTopicsOverviewQuery(query)) {
      const bundle = await fetchTopicsBundle();
      let cards: GroveeNewsCard[] = bundle.cards.slice(0, 24);
      if (!cards.length && stats.rssHeadlines > 0) {
        const hits = await buildRecentHeadlineHits(query, 24);
        cards = await hitsToDisplayCards(hits);
      }
      const text =
        cards.length > 0
          ? formatHeadlinesForBrief(cards)
          : `מאגר מקומי: ${stats.rssHeadlines} כותרות — Topics עדיין מתמלא`;

      return {
        provider: PROVIDER,
        label: `חדשות (Topics · ${bundle.stats.lanesWithHits || cards.length} נושאים)`,
        ok: cards.length > 0,
        text,
        newsCards: cards,
        error: cards.length ? undefined : "אין עדיין נושאים מוכנים — המתן לאיסוף ברקע",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    let hits = await searchNewsWithFallback(query);
    if (heUi && hits.length < 8) {
      await priorityPollHebrewFeeds({ timeoutMs: 15_000 });
      const refreshed = await searchNewsWithFallback(query);
      if (refreshed.length > hits.length) hits = refreshed;
    }

    const cards = await hitsToDisplayCards(hits);
    const result = buildNewsSearchResult(query, cards, stats, started, label);

    if (heUi && options?.onRefresh) {
      void continueHebrewNewsScan(query, hits.length, options.onRefresh);
    }

    return result;
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
