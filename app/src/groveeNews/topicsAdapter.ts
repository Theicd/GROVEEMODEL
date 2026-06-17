import { dedupeNewsCards } from "./dedupeCards";
import { applyDisplayLanguageBatch } from "./engine/display/liveFeedDisplay";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import { buildTopicsDigest } from "./engine/topics/buildTopicsDigest";
import { TOPICS_PER_LANE } from "./engine/topics/topicLanes";
import type { GroveeTopicCard, GroveeTopicsBundle } from "./types";

export async function fetchTopicsBundle(): Promise<GroveeTopicsBundle> {
  const digest = await buildTopicsDigest({ perLane: TOPICS_PER_LANE });
  const profile = getUserNewsProfile();

  let articles = digest.hits.map((h) => h.article);
  if (profile.uiLanguage !== "en") {
    articles = await applyDisplayLanguageBatch(articles, profile.uiLanguage);
  }

  const cards: GroveeTopicCard[] = digest.hits.map((hit, i) => {
    const a = articles[i] ?? hit.article;
    return {
      id: hit.id,
      title: a.displayTitle || a.title,
      titleOriginal: a.title,
      source: a.source,
      sourceKey: a.sourceKey,
      url: a.url,
      image: a.image,
      score: hit.score,
      publishedTs: a.publishedTs,
      summary: a.displaySummary || a.summary,
      laneId: hit.laneId,
      laneLabel: hit.laneLabel,
      laneIcon: hit.laneIcon,
      query: hit.query,
      matchLabel: hit.matchLabel,
    };
  });

  return {
    generatedAt: digest.generatedAt,
    cards: dedupeNewsCards(cards),
    stats: {
      totalLanes: digest.stats.totalLanes,
      lanesWithHits: digest.stats.lanesWithHits,
    },
  };
}
