import { queryHasHebrew } from "./hebrewSearchTerms";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import { getAllRssItems } from "./engine/storage/db";
import { rankRssHeadlinesForQuery, rssItemToSearchArticle, isTopHitRelevant } from "./engine/search/relevance";
import type { SearchHit } from "./engine/types";
import {
  extractNewsTopicTerms,
  isBroadNewsOverviewQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import { isNewsQuery } from "../webSearch/intents";

/** Fallback when lexical search returns nothing but RSS DB has headlines. */
export async function buildRecentHeadlineHits(query: string, limit = 12): Promise<SearchHit[]> {
  const rssItems = await getAllRssItems(800);
  if (!rssItems.length) return [];

  const engineQuery = normalizeNewsEngineQuery(query);
  const terms = extractNewsTopicTerms(query);
  const specific = isSpecificNewsTopicQuery(query);

  if (specific && !terms.length && !queryHasHebrew(query)) return [];

  if (specific && (engineQuery || queryHasHebrew(query))) {
    const rankQuery = engineQuery || query;
    const ranked = rankRssHeadlinesForQuery(rssItems, rankQuery, new Set(), limit * 3);
    const byId = new Map(rssItems.map((r) => [r.id, r]));
    const hits = ranked
      .map((h) => {
        const item = byId.get(h.id);
        if (!item) return null;
        const article = rssItemToSearchArticle(item);
        if (!isTopHitRelevant(article, query)) return null;
        return {
          article,
          cluster: null,
          score: h.score,
          sourceKind: "headline" as const,
        } satisfies SearchHit;
      })
      .filter((h) => h !== null)
      .slice(0, limit);
    if (hits.length) return hits;
  }

  const rankQuery = engineQuery || query.trim() || "news";
  const ranked = rankRssHeadlinesForQuery(rssItems, rankQuery, new Set(), limit);
  if (ranked.length) {
    const byId = new Map(rssItems.map((r) => [r.id, r]));
    const rankedHits = ranked
      .map((h) => {
        const item = byId.get(h.id);
        if (!item) return null;
        return {
          article: rssItemToSearchArticle(item),
          cluster: null,
          score: h.score,
          sourceKind: "headline" as const,
        } satisfies SearchHit;
      })
      .filter((h) => h !== null);
    if (rankedHits.length) return rankedHits;
  }

  if (isBroadNewsOverviewQuery(engineQuery) || isSpecificNewsTopicQuery(query) || isNewsQuery(query)) {
    /* fall through to recent headlines */
  } else if (!rankQuery || rankQuery.length < 2) {
    return [];
  }

  const preferIlHe = getUserNewsProfile().uiLanguage === "he";
  const sorted = [...rssItems].sort((a, b) => {
    if (preferIlHe) {
      const aIl = (a.sourceKey ?? "").startsWith("il_") ? 1 : 0;
      const bIl = (b.sourceKey ?? "").startsWith("il_") ? 1 : 0;
      if (aIl !== bIl) return bIl - aIl;
    }
    return b.publishedTs - a.publishedTs;
  });
  return sorted.slice(0, limit).map((item) => ({
    article: rssItemToSearchArticle(item),
    cluster: null,
    score: 8,
    sourceKind: "headline" as const,
  }));
}
