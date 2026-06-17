import { dedupeNewsCards } from "./dedupeCards";
import { applyDisplayLanguageBatch } from "./engine/display/liveFeedDisplay";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import type { SearchHit } from "./engine/types";
import type { GroveeNewsCard } from "./types";

const MAX_CARDS = 12;

export function hitToCard(hit: SearchHit, article = hit.article): GroveeNewsCard {
  return {
    id: article.id || article.url,
    title: article.displayTitle || article.title,
    titleOriginal: article.title,
    source: article.source,
    sourceKey: article.sourceKey,
    url: article.url,
    image: article.image,
    score: hit.score,
    publishedTs: article.publishedTs,
    summary: article.displaySummary || article.summary,
  };
}

export async function hitsToDisplayCards(hits: SearchHit[]): Promise<GroveeNewsCard[]> {
  const sliced = hits.slice(0, MAX_CARDS);
  if (!sliced.length) return [];

  const profile = getUserNewsProfile();
  let articles = sliced.map((h) => h.article);
  if (profile.uiLanguage !== "en") {
    articles = await applyDisplayLanguageBatch(articles, profile.uiLanguage);
  }
  return dedupeNewsCards(sliced.map((hit, i) => hitToCard(hit, articles[i])));
}
