// @ts-nocheck
import { getArticlesSince } from "../storage/db";
import { hasDisplayableContent } from "../summarize/summaryQuality";
import type { ArticleFeedItem } from "../feed/buildFeed";
import { articleTimestamp } from "../feed/time";
import { pickUniqueSourceArticles } from "../feed/uniqueSourcePick";

const BRIEFING_WINDOW_MS = 24 * 60 * 60 * 1000;

/** Balanced briefing: up to `limit` stories from distinct RSS feeds (one per sourceKey). */
export async function buildDailyBriefing(limit = 20): Promise<ArticleFeedItem[]> {
  const cutoff = Date.now() - BRIEFING_WINDOW_MS;
  const recent = (await getArticlesSince(cutoff, limit * 12)).filter((a) => hasDisplayableContent(a));
  const picked = pickUniqueSourceArticles(recent, limit);

  return picked
    .map((article) => ({
      kind: "article" as const,
      id: article.id,
      article,
      sortTs: articleTimestamp(article),
    }))
    .sort((a, b) => b.sortTs - a.sortTs);
}
