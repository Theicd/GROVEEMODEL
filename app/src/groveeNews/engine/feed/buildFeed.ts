// @ts-nocheck
import { isDeepReadEnabled } from "../engine/deepReadGate";
import { applyDisplayLanguageBatch } from "../display/liveFeedDisplay";
import { getAllRssItems, getMultiSourceClusters, getSummarizedArticles } from "../storage/db";
import { rssItemToSearchArticle } from "../search/relevance";
import { hasDisplayableContent } from "../summarize/summaryQuality";
import type { ArticleRecord, StoryCluster } from "../types";
import { articleTimestamp } from "./time";

export type TrendingFeedItem = {
  kind: "trending";
  id: string;
  cluster: StoryCluster;
  articles: ArticleRecord[];
  lead: ArticleRecord;
  mergedFacts: string[];
  popularity: number;
  sortTs: number;
};

export type ArticleFeedItem = {
  kind: "article";
  id: string;
  article: ArticleRecord;
  sortTs: number;
};

export type FeedItem = TrendingFeedItem | ArticleFeedItem;

import { latestRssPerSource } from "./mixRssBySource";
import { slicePage } from "./buildFeedPagination";

export type BuildFeedPageOptions = {
  pageSize?: number;
  offset?: number;
};

const CONFIDENCE_SCORE: Record<StoryCluster["confidence"], number> = {
  HIGH: 30,
  MEDIUM: 18,
  LOW: 8,
};

const DEFAULT_PAGE_SIZE = 30;
/** Live tab shows exactly this many newest headlines — no infinite scroll. */
export const LIVE_FEED_LIMIT = 30;
const MAX_TRENDING_ON_FIRST_PAGE = 6;

function groupByCluster(articles: ArticleRecord[]): Map<string, ArticleRecord[]> {
  const map = new Map<string, ArticleRecord[]>();
  for (const a of articles) {
    if (!a.clusterId) continue;
    const list = map.get(a.clusterId) ?? [];
    list.push(a);
    map.set(a.clusterId, list);
  }
  return map;
}

function clusterSortTs(articles: ArticleRecord[]): number {
  return Math.max(0, ...articles.map(articleTimestamp));
}

export type FeedPage = {
  items: FeedItem[];
  hasMore: boolean;
  nextOffset: number;
};

export { slicePage } from "./buildFeedPagination";

/** Live tab (headlines mode): 30 newest headlines globally — fixed window, no pagination. */
export async function buildRssFeedPage(options: BuildFeedPageOptions = {}): Promise<FeedPage> {
  const limit = Math.min(options.pageSize ?? LIVE_FEED_LIMIT, LIVE_FEED_LIMIT);

  const perSource = latestRssPerSource(await getAllRssItems(2500));
  const page = perSource.slice(0, limit);

  const rawArticles = page.map((item) => rssItemToSearchArticle(item));
  const translated = await applyDisplayLanguageBatch(rawArticles);

  const items: ArticleFeedItem[] = translated.map((article, i) => ({
    kind: "article",
    id: article.id,
    article,
    sortTs: page[i].publishedTs,
  }));

  return {
    items,
    hasMore: false,
    nextOffset: limit,
  };
}

/** Summarized articles + trending clusters — paginated window. */
export async function buildSummarizedFeedPage(options: BuildFeedPageOptions = {}): Promise<FeedPage> {
  const pageSize = options.pageSize ?? DEFAULT_PAGE_SIZE;
  const offset = options.offset ?? 0;

  // Fetch enough summarized rows to fill trending + page after offset.
  const scanLimit = Math.min(offset + pageSize + 80, 400);
  const summarized = await getSummarizedArticles(scanLimit);
  const byCluster = groupByCluster(summarized);
  const items: FeedItem[] = [];
  const usedIds = new Set<string>();

  if (offset === 0) {
    const multi = await getMultiSourceClusters();
    const sortedClusters = [...multi]
      .sort((a, b) => {
        const sa = b.sourceKeys.length * 10 + CONFIDENCE_SCORE[b.confidence];
        const sb = a.sourceKeys.length * 10 + CONFIDENCE_SCORE[a.confidence];
        return sa - sb;
      })
      .slice(0, MAX_TRENDING_ON_FIRST_PAGE);

    for (const cluster of sortedClusters) {
      const articles = (byCluster.get(cluster.id) ?? [])
        .filter((a) => a.summary?.trim() && hasDisplayableContent(a));
      if (!articles.length) continue;

      const lead =
        articles.find((a) => a.image && articleTimestamp(a) === clusterSortTs(articles)) ??
        articles.find((a) => a.image) ??
        articles.find((a) => a.summarizedAt > 0) ??
        articles[0];

      items.push({
        kind: "trending",
        id: cluster.id,
        cluster,
        articles,
        lead,
        mergedFacts: [...new Set(articles.flatMap((a) => a.keyFacts))].slice(0, 4),
        popularity: cluster.sourceKeys.length * 10 + CONFIDENCE_SCORE[cluster.confidence],
        sortTs: clusterSortTs(articles),
      });

      for (const a of articles) usedIds.add(a.id);
    }
  }

  for (const article of summarized) {
    if (usedIds.has(article.id)) continue;
    if (!hasDisplayableContent(article)) continue;
    items.push({
      kind: "article",
      id: article.id,
      article,
      sortTs: articleTimestamp(article),
    });
  }

  items.sort((a, b) => b.sortTs - a.sortTs);
  return slicePage(items, offset, pageSize);
}

/** Live tab: RSS when Deep Read off, summarized when on — always capped at LIVE_FEED_LIMIT. */
export async function buildLiveFeedPage(options: BuildFeedPageOptions = {}): Promise<FeedPage> {
  const capped = { ...options, pageSize: LIVE_FEED_LIMIT, offset: 0 };
  if (!isDeepReadEnabled()) {
    return buildRssFeedPage(capped);
  }
  const page = await buildSummarizedFeedPage(capped);
  return {
    ...page,
    items: page.items.slice(0, LIVE_FEED_LIMIT),
    hasMore: false,
    nextOffset: LIVE_FEED_LIMIT,
  };
}

/** @deprecated use buildLiveFeedPage */
export async function buildNewsFeed(articleLimit = 150): Promise<FeedItem[]> {
  const page = await buildSummarizedFeedPage({ pageSize: articleLimit, offset: 0 });
  return page.items;
}
