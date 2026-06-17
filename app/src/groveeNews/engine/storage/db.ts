// @ts-nocheck
import Dexie, { type Table } from "dexie";
import { isBlockedArticle, isBlockedRssItem } from "../feeds/blockedFeeds";
import { hasImageUrl, hasRealImageUrl, isStockImageUrl, normalizeImageUrl } from "../media/imageFields";
import type { ArticleRecord, RssItem, StoryCluster } from "../types";

export class NewsDb extends Dexie {
  rssItems!: Table<RssItem, string>;
  articles!: Table<ArticleRecord, string>;
  clusters!: Table<StoryCluster, string>;
  meta!: Table<{ key: string; value: string }, string>;

  constructor() {
    super("grovee-news-intel");
    this.version(1).stores({
      rssItems: "id, publishedTs, sourceKey, link",
      articles: "id, publishedTs, sourceKey, clusterId, url",
      clusters: "id, updatedAt",
      meta: "key",
    });
  }
}

export const db = new NewsDb();

export async function upsertRssItems(items: RssItem[]): Promise<number> {
  let n = 0;
  const allowed = items.filter((item) => !isBlockedRssItem(item));
  await db.transaction("rw", db.rssItems, async () => {
    for (const item of allowed) {
      const exists = await db.rssItems.get(item.id);
      if (!exists) {
        await db.rssItems.put(item);
        n++;
      }
    }
  });
  return n;
}

export async function upsertArticle(article: ArticleRecord): Promise<void> {
  await db.articles.put(article);
}

export async function patchArticleImage(id: string, image: string): Promise<void> {
  const url = normalizeImageUrl(image);
  if (!id || !url || isStockImageUrl(url)) return;

  const article = await db.articles.get(id);
  if (article) {
    if (!hasRealImageUrl(article.image)) {
      await db.articles.put({ ...article, image: url });
    }
    return;
  }

  const rss = await db.rssItems.get(id);
  if (rss && !hasRealImageUrl(rss.image)) {
    await db.rssItems.put({ ...rss, image: url });
  }
}

export async function upsertCluster(cluster: StoryCluster): Promise<void> {
  await db.clusters.put(cluster);
}

export async function getArticleCount(): Promise<number> {
  return db.articles.count();
}

export async function getAllArticles(limit = 5000): Promise<ArticleRecord[]> {
  const articles = await db.articles.orderBy("publishedTs").reverse().limit(limit).toArray();
  return articles.filter((a) => !isBlockedArticle(a));
}

export async function getAllClusters(): Promise<StoryCluster[]> {
  return db.clusters.orderBy("updatedAt").reverse().toArray();
}

export async function getMultiSourceClusters(): Promise<StoryCluster[]> {
  const all = await getAllClusters();
  return all.filter((c) => c.sourceKeys.length > 1);
}

export async function getCluster(id: string): Promise<StoryCluster | undefined> {
  return db.clusters.get(id);
}

export async function getArticlesByCluster(clusterId: string): Promise<ArticleRecord[]> {
  return db.articles.where("clusterId").equals(clusterId).toArray();
}

export async function setMeta(key: string, value: string): Promise<void> {
  await db.meta.put({ key, value });
}

export async function getMeta(key: string): Promise<string | null> {
  const row = await db.meta.get(key);
  return row?.value ?? null;
}

export async function getRssHeadlineCount(): Promise<number> {
  return db.rssItems.count();
}

export async function getSummarizedCount(): Promise<number> {
  return db.articles.filter((a) => a.summarizedAt > 0).count();
}

export async function getPendingArticleCount(): Promise<number> {
  const [articleUrls, rssItems] = await Promise.all([
    getRecentArticleUrls(4000),
    getAllRssItems(600),
  ]);
  return rssItems.filter((r) => !articleUrls.has(r.link)).length;
}

export async function getRecentArticles(limit = 8): Promise<ArticleRecord[]> {
  const batch = await db.articles.orderBy("publishedTs").reverse().limit(limit * 4).toArray();
  return batch.filter((a) => !isBlockedArticle(a)).slice(0, limit);
}

/** Recent article URLs for pending-RSS checks (bounded memory). */
export async function getRecentArticleUrls(cap = 4000): Promise<Set<string>> {
  const batch = await db.articles.orderBy("publishedTs").reverse().limit(cap).toArray();
  return new Set(batch.map((a) => a.url));
}

/** Articles published after cutoff, newest first (bounded scan). */
export async function getArticlesSince(cutoffTs: number, limit = 120): Promise<ArticleRecord[]> {
  const batch = await db.articles.orderBy("publishedTs").reverse().limit(limit * 6).toArray();
  const out: ArticleRecord[] = [];
  for (const a of batch) {
    if (isBlockedArticle(a)) continue;
    const ts = a.publishedTs || a.summarizedAt || a.fetchedAt || 0;
    if (ts < cutoffTs) break;
    out.push(a);
    if (out.length >= limit) break;
  }
  return out;
}

/**
 * Articles with summaries, newest first — paginated without loading the full table.
 * Scans publishedTs index in batches until enough matches.
 */
export async function getSummarizedArticles(limit = 30, skip = 0): Promise<ArticleRecord[]> {
  const result: ArticleRecord[] = [];
  let skipped = 0;
  let offset = 0;
  const batchSize = 120;

  while (result.length < limit) {
    const batch = await db.articles.orderBy("publishedTs").reverse().offset(offset).limit(batchSize).toArray();
    if (!batch.length) break;
    offset += batch.length;

    for (const a of batch) {
      if (isBlockedArticle(a)) continue;
      if (!a.summary?.trim()) continue;
      if (skipped < skip) {
        skipped++;
        continue;
      }
      result.push(a);
      if (result.length >= limit) return result;
    }

    if (batch.length < batchSize) break;
  }
  return result;
}

export async function getAllRssItems(limit = 5000): Promise<RssItem[]> {
  const items = await db.rssItems.orderBy("publishedTs").reverse().limit(limit).toArray();
  return items.filter((item) => !isBlockedRssItem(item));
}

/** Remove blocked outlets (e.g. Al Jazeera) from local storage. */
export async function purgeBlockedNewsSources(): Promise<{ articles: number; rss: number }> {
  const articles = (await db.articles.toArray()).filter(isBlockedArticle);
  const rss = (await db.rssItems.toArray()).filter(isBlockedRssItem);
  const removedArticleIds = new Set(articles.map((a) => a.id));

  await db.transaction("rw", db.articles, db.rssItems, db.clusters, async () => {
    for (const a of articles) await db.articles.delete(a.id);
    for (const r of rss) await db.rssItems.delete(r.id);

    const clusters = await db.clusters.toArray();
    for (const cluster of clusters) {
      const keptIds = cluster.articleIds.filter((id) => !removedArticleIds.has(id));
      if (keptIds.length === 0) {
        await db.clusters.delete(cluster.id);
        continue;
      }
      if (keptIds.length !== cluster.articleIds.length) {
        const keptArticles = await db.articles.bulkGet(keptIds);
        const sourceKeys = [...new Set(keptArticles.filter(Boolean).map((a) => a!.sourceKey))];
        await db.clusters.put({
          ...cluster,
          articleIds: keptIds,
          sourceKeys,
        });
      }
    }
  });

  return { articles: articles.length, rss: rss.length };
}

export async function getRssItemByLink(link: string): Promise<RssItem | undefined> {
  if (!link) return undefined;
  const hit = await db.rssItems.where("link").equals(link).first();
  return hit && !isBlockedRssItem(hit) ? hit : undefined;
}

/** Recent articles missing a hero image (for slow backfill). */
export async function listArticlesMissingImages(limit = 20): Promise<ArticleRecord[]> {
  const batch = await db.articles.orderBy("publishedTs").reverse().limit(limit * 8).toArray();
  const out: ArticleRecord[] = [];
  for (const a of batch) {
    if (isBlockedArticle(a)) continue;
    if (hasRealImageUrl(a.image)) continue;
    out.push(a);
    if (out.length >= limit) break;
  }
  return out;
}

export async function listPendingRssItems(limit = 30): Promise<RssItem[]> {
  const articleUrls = await getRecentArticleUrls(4000);
  const all = await getAllRssItems(1200);
  const pending = all.filter((r) => !articleUrls.has(r.link));

  // Round-robin by category so tech / TCM / dev are not starved by world-news volume
  const buckets = new Map<string, RssItem[]>();
  for (const item of pending) {
    const cat = item.category || "world";
    const list = buckets.get(cat) ?? [];
    list.push(item);
    buckets.set(cat, list);
  }
  for (const [cat, list] of buckets) {
    list.sort((a, b) => b.publishedTs - a.publishedTs);
    buckets.set(cat, list);
  }

  const categories = [...buckets.keys()].sort();
  const picked: RssItem[] = [];
  let round = 0;
  while (picked.length < limit) {
    let any = false;
    for (const cat of categories) {
      const list = buckets.get(cat)!;
      if (round < list.length) {
        picked.push(list[round]);
        any = true;
        if (picked.length >= limit) break;
      }
    }
    if (!any) break;
    round++;
  }
  return picked;
}
