// @ts-nocheck
import { db, getPendingArticleCount } from "../storage/db";
import { isBlockedArticle, isBlockedRssItem } from "../feeds/blockedFeeds";
import { hasImageUrl } from "../media/imageFields";
import type { EngineLibraryStats, LanguageStat } from "../types";
const SAMPLE = 2000;

export function countLanguages(articles: { language?: string }[]): LanguageStat[] {
  const map = new Map<string, number>();
  for (const a of articles) {
    const code = (a.language || "multi").slice(0, 12);
    map.set(code, (map.get(code) ?? 0) + 1);
  }
  return [...map.entries()]
    .map(([code, count]) => ({ code, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
}

async function countUniqueSourceKeys(limit = 4000): Promise<number> {
  const items = await db.rssItems.orderBy("publishedTs").reverse().limit(limit).toArray();
  return new Set(items.filter((r) => !isBlockedRssItem(r)).map((r) => r.sourceKey)).size;
}

/** Snapshot of local IndexedDB + search index (independent of AI model). */
export async function getEngineLibraryStats(
  searchIndexSize: number,
): Promise<Omit<EngineLibraryStats, "searchIndexSize">> {
  const [rssHeadlines, articlesIndexed, pendingArticles, summarizedByModel, rssWithImages, articlesWithImages, uniqueRssSources] =
    await Promise.all([
      db.rssItems.count(),
      db.articles.count(),
      getPendingArticleCount(),
      db.articles.filter((a) => a.summarizedAt > 0).count(),
      db.rssItems.filter((r) => hasImageUrl(r.image)).count(),
      db.articles.filter((a) => hasImageUrl(a.image)).count(),      countUniqueSourceKeys(),
    ]);

  const [rssSample, articleSample] = await Promise.all([
    db.rssItems.orderBy("publishedTs").reverse().limit(SAMPLE).toArray(),
    db.articles.orderBy("publishedTs").reverse().limit(SAMPLE).toArray(),
  ]);

  const artClean = articleSample.filter((a) => !isBlockedArticle(a));

  return {
    rssHeadlines,
    rssWithImages,
    rssWithoutImages: Math.max(0, rssHeadlines - rssWithImages),
    articlesIndexed,
    articlesWithImages,
    articlesWithoutImages: Math.max(0, articlesIndexed - articlesWithImages),
    pendingArticles,
    summarizedByModel,
    uniqueRssSources,
    uniqueFeedSourcesInArticles: new Set(artClean.map((a) => a.sourceKey)).size,
    languages: countLanguages(artClean.length ? artClean : rssSample.filter((r) => !isBlockedRssItem(r))),
    sampledAt: Date.now(),
  };
}
