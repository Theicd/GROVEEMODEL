// @ts-nocheck
import type { ArticleRecord } from "../types";
import { articleTimestamp } from "./time";

/** Keep the newest article per RSS feed key (`sourceKey`). */
export function newestPerSourceKey(articles: ArticleRecord[]): ArticleRecord[] {
  const sorted = [...articles].sort((a, b) => articleTimestamp(b) - articleTimestamp(a));
  const byKey = new Map<string, ArticleRecord>();
  for (const a of sorted) {
    if (!byKey.has(a.sourceKey)) byKey.set(a.sourceKey, a);
  }
  return [...byKey.values()];
}

function categoryPriority(cat: string): number {
  if (cat === "world") return 0;
  if (cat === "technology" || cat === "ai" || cat === "dev") return 1;
  return 2;
}

/**
 * Pick up to `limit` stories from distinct RSS sources (one per sourceKey).
 * Round-robin across feed categories for a balanced mix.
 */
export function pickUniqueSourceArticles(articles: ArticleRecord[], limit: number): ArticleRecord[] {
  const pool = newestPerSourceKey(articles);
  if (pool.length <= limit) {
    return pool.sort((a, b) => articleTimestamp(b) - articleTimestamp(a));
  }

  const buckets = new Map<string, ArticleRecord[]>();
  for (const article of pool) {
    const cat = article.feedCategory || article.sourceKey || "world";
    const list = buckets.get(cat) ?? [];
    list.push(article);
    buckets.set(cat, list);
  }

  for (const [cat, list] of buckets) {
    list.sort((a, b) => articleTimestamp(b) - articleTimestamp(a));
    buckets.set(cat, list);
  }

  const categories = [...buckets.keys()].sort((a, b) => categoryPriority(a) - categoryPriority(b));
  const picked: ArticleRecord[] = [];
  const usedSources = new Set<string>();
  let round = 0;

  while (picked.length < limit) {
    let any = false;
    for (const cat of categories) {
      const list = buckets.get(cat)!;
      if (round < list.length) {
        const article = list[round];
        if (!usedSources.has(article.sourceKey)) {
          picked.push(article);
          usedSources.add(article.sourceKey);
          any = true;
          if (picked.length >= limit) break;
        }
      }
    }
    if (!any) break;
    round++;
  }

  return picked.sort((a, b) => articleTimestamp(b) - articleTimestamp(a));
}

export function countUniqueSourceKeys(articles: { sourceKey: string }[]): number {
  return new Set(articles.map((a) => a.sourceKey)).size;
}
