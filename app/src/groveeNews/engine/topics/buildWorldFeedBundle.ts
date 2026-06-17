// @ts-nocheck
import { getFeedLang } from "../feeds/feedRegistry";
import { getUserNewsProfile, isRtlLanguage, type UserNewsProfile } from "../settings/userNewsProfile";
import { translateTexts, type TranslateProvider } from "../translate/googleTranslate";
import { buildTopicsDigest, type TopicDigestHit } from "./buildTopicsDigest";
import { yieldToMain } from "../util/yieldToMain";

export const WORLD_FEED_SLOTS = 30;

export type WorldFeedCard = {
  slot: number;
  id: string;
  laneId: string;
  laneLabel: string;
  laneLabelDisplay: string;
  laneIcon: string;
  title: string;
  titleOriginal: string;
  url: string;
  image: string;
  source: string;
  query: string;
  feedLang: string;
  rtl: boolean;
};

export type WorldFeedBundle = {
  generatedAt: number;
  cards: WorldFeedCard[];
  translateProvider: TranslateProvider | "none";
  uiLanguage: string;
};

function mergeProfile(override?: Partial<UserNewsProfile>): UserNewsProfile {
  return { ...getUserNewsProfile(), ...override };
}

export function needsTranslation(feedLang: string, uiLang: string): boolean {
  if (uiLang === "en") return feedLang !== "en" && feedLang !== "multi";
  return feedLang !== uiLang;
}

function hitToCard(hit: TopicDigestHit, slot: number): WorldFeedCard {
  const feedLang = getFeedLang(hit.article.sourceKey);
  return {
    slot,
    id: hit.id,
    laneId: hit.laneId,
    laneLabel: hit.laneLabel,
    laneLabelDisplay: hit.laneLabel,
    laneIcon: hit.laneIcon,
    title: hit.article.title,
    titleOriginal: hit.article.title,
    url: hit.article.url,
    image: hit.article.image,
    source: hit.article.source,
    query: hit.query,
    feedLang,
    rtl: isRtlLanguage(feedLang),
  };
}

/** Build 30-card world mix from global topic digest (no regional slots). */
export async function buildWorldFeedCards(
  override?: Partial<UserNewsProfile>,
): Promise<Omit<WorldFeedBundle, "translateProvider"> & { translateProvider: "none" }> {
  const profile = mergeProfile(override);
  const uiLang = profile.uiLanguage;

  const digest = await buildTopicsDigest({ perLane: 1, lightMix: true });
  await yieldToMain();

  const seen = new Set<string>();
  const cards: WorldFeedCard[] = [];

  for (const hit of digest.hits) {
    if (cards.length >= WORLD_FEED_SLOTS) break;
    const url = hit.article.url;
    if (!url || seen.has(url)) continue;
    seen.add(url);
    cards.push(hitToCard(hit, cards.length + 1));
  }

  cards.forEach((c, i) => {
    c.slot = i + 1;
  });

  const rtl = isRtlLanguage(uiLang);
  cards.forEach((c) => {
    c.rtl = rtl || isRtlLanguage(c.feedLang);
  });

  return {
    generatedAt: digest.generatedAt,
    cards,
    translateProvider: "none",
    uiLanguage: uiLang,
  };
}

/** Apply Google Translate to titles + lane labels where feed lang ≠ ui lang. */
export async function applyWorldFeedTranslations(
  cards: WorldFeedCard[],
  uiLang: string,
  onProgress?: () => void,
): Promise<TranslateProvider | "none"> {
  const titleIndices: number[] = [];
  const laneIndices: number[] = [];
  const titles: string[] = [];
  const lanes: string[] = [];

  cards.forEach((card, i) => {
    if (needsTranslation(card.feedLang, uiLang)) {
      titleIndices.push(i);
      titles.push(card.titleOriginal);
    }
    if (uiLang !== "en" && card.laneLabel) {
      laneIndices.push(i);
      lanes.push(card.laneLabel);
    }
  });

  const uniqueTexts = [...titles, ...lanes];
  if (!uniqueTexts.length) return "none";

  const { texts: translated, provider } = await translateTexts(uniqueTexts, uiLang);
  onProgress?.();

  titles.forEach((_, idx) => {
    const card = cards[titleIndices[idx]];
    if (card) card.title = translated[idx] ?? card.title;
  });
  const laneOffset = titles.length;
  lanes.forEach((_, idx) => {
    const card = cards[laneIndices[idx]];
    if (card) card.laneLabelDisplay = translated[laneOffset + idx] ?? card.laneLabel;
  });

  return provider;
}

export async function buildWorldFeedBundle(
  override?: Partial<UserNewsProfile>,
  onPartial?: (cards: WorldFeedCard[]) => void,
): Promise<WorldFeedBundle> {
  const partial = await buildWorldFeedCards(override);
  onPartial?.(partial.cards);

  const provider = await applyWorldFeedTranslations(partial.cards, partial.uiLanguage, () => {
    onPartial?.(partial.cards);
  });

  return { ...partial, translateProvider: provider };
}
