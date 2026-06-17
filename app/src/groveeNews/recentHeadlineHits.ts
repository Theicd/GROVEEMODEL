import { getAllRssItems } from "./engine/storage/db";
import { rankRssHeadlinesForQuery, rssItemToSearchArticle, isTopHitRelevant } from "./engine/search/relevance";
import type { SearchHit } from "./engine/types";
import {
  extractNewsTopicTerms,
  isBroadNewsOverviewQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";

/** Fallback when lexical search returns nothing but RSS cache has headlines. */
export async function buildRecentHeadlineHits(query: string, limit = 12): Promise<SearchHit[]> {
  const rssItems = await getAllRssItems(800);
  if (!rssItems.length) return [];

  const engineQuery = normalizeNewsEngineQuery(query);
  const terms = extractNewsTopicTerms(query);
  const specific = isSpecificNewsTopicQuery(query);

  if (specific && !terms.length) return [];

  if (specific && engineQuery) {
    const ranked = rankRssHeadlinesForQuery(rssItems, engineQuery, new Set(), limit * 3);
    const byId = new Map(rssItems.map((r) => [r.id, r]));
    const hits = ranked
      .map((h) => {
        const item = byId.get(h.id);
        if (!item) return null;
        const article = rssItemToSearchArticle(item);
        if (!isTopHitRelevant(article, engineQuery)) return null;
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
    return [];
  }

  if (!isBroadNewsOverviewQuery(engineQuery)) return [];

  const ranked = rankRssHeadlinesForQuery(rssItems, engineQuery || "world news", new Set(), limit);
  if (ranked.length) {
    const byId = new Map(rssItems.map((r) => [r.id, r]));
    return ranked
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
  }

  const sorted = [...rssItems].sort((a, b) => b.publishedTs - a.publishedTs);
  return sorted.slice(0, limit).map((item) => ({
    article: rssItemToSearchArticle(item),
    cluster: null,
    score: 8,
    sourceKind: "headline" as const,
  }));
}
