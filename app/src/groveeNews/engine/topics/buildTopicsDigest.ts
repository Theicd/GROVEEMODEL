// @ts-nocheck
import { applyDisplayLanguageBatch } from "../display/liveFeedDisplay";
import { getAllArticles, getAllRssItems } from "../storage/db";
import {
  rankRssHeadlinesForQuery,
  rssItemToSearchArticle,
} from "../search/relevance";
import { rankIndexedArticlesForQuery } from "../search/flexIndex";
import { hasImageUrl } from "../media/imageFields";
import { extractImageFromDescription } from "../media/imageResolve";
import type { ArticleRecord, RssItem } from "../types";
import { yieldToMain } from "../util/yieldToMain";
import { TOPIC_LANES, TOPICS_PER_LANE, type TopicLane } from "./topicLanes";

export type TopicDigestHit = {
  id: string;
  laneId: string;
  laneLabel: string;
  laneIcon: string;
  query: string;
  score: number;
  matchLabel: "high" | "medium" | "low";
  article: ArticleRecord;
  sourceKind: "indexed" | "headline";
  /** True when filled from category / pool fallback (not keyword rank). */
  fallback?: boolean;
};

export type TopicsDigest = {
  generatedAt: number;
  hits: TopicDigestHit[];
  stats: {
    totalLanes: number;
    lanesWithHits: number;
    keywordHits: number;
    fallbackHits: number;
    headlineHits: number;
    indexedHits: number;
  };
};

export type BuildTopicsDigestOptions = {
  /** Top picks per lane (default 1 for a clean mix). */
  perLane?: number;
  lanes?: TopicLane[];
  /** Boost headlines from these feed keys. */
  boostSourceKeys?: Set<string>;
  /** Lighter digest — still tries to fill every lane. */
  lightMix?: boolean;
};

/** Feed categories that best match a topic lane when keyword search is empty. */
const LANE_FEED_CATEGORIES: Partial<Record<string, string[]>> = {
  israel: ["israel", "world"],
  ukraine: ["world"],
  russia: ["world"],
  china: ["world", "business"],
  iran: ["world"],
  turkey: ["world"],
  africa: ["world"],
  india: ["world", "business"],
  latam: ["world", "business"],
  war: ["world", "israel"],
  diplomacy: ["world"],
  ai: ["ai", "technology", "dev"],
  tech: ["technology", "dev"],
  cyber: ["technology", "dev"],
  startup: ["technology", "dev", "business"],
  robotics: ["technology", "ai"],
  biotech: ["science", "health"],
  space: ["science", "space", "world"],
  science: ["science"],
  nuclear: ["science", "world"],
  market: ["business"],
  crypto: ["business", "technology"],
  energy: ["business", "world"],
  car: ["technology", "business"],
  aviation: ["world", "business"],
  maritime: ["business", "world"],
  politics: ["world"],
  crime: ["world"],
  health: ["health"],
  climate: ["science", "world"],
  environment: ["science", "world"],
  education: ["world"],
  religion: ["world"],
  travel: ["food", "world"],
  food: ["food"],
  fashion: ["fashion", "entertainment"],
  music: ["entertainment"],
  film: ["entertainment"],
  gaming: ["entertainment"],
  sport: ["sport"],
  tcm: ["health", "alternative"],
};

function rssHasPreviewImage(item: RssItem): boolean {
  if (hasImageUrl(item.image)) return true;
  return Boolean(extractImageFromDescription(item.description, item.link));
}

function articleHasPreviewImage(article: ArticleRecord): boolean {
  if (hasImageUrl(article.image)) return true;
  return Boolean(extractImageFromDescription(article.articleText, article.url));
}

const IMAGE_PICK_BOOST = 28;

function articleFromRss(item: RssItem): ArticleRecord {
  return rssItemToSearchArticle(item);
}

function hitFromRss(
  lane: TopicLane,
  item: RssItem,
  score: number,
  matchLabel: TopicDigestHit["matchLabel"],
  fallback: boolean,
): TopicDigestHit {
  return {
    id: `${lane.id}:headline:${item.id}`,
    laneId: lane.id,
    laneLabel: lane.label,
    laneIcon: lane.icon,
    query: lane.query,
    score,
    matchLabel,
    article: articleFromRss(item),
    sourceKind: "headline",
    fallback,
  };
}

const LOOSE_QUERY_PATTERNS: Record<string, RegExp> = {
  ai: /\b(ai|artificial intelligence|machine learning|llm|gpt|openai|deepmind)\b/i,
  car: /\b(car|ev|vehicle|automotive|tesla|byd)\b/i,
  tcm: /\b(acupuncture|herbal|traditional medicine|holistic|tcm)\b/i,
  crypto: /\b(crypto|bitcoin|ethereum|blockchain|defi|nft)\b/i,
  latam: /\b(brazil|mexico|argentina|chile|colombia|latin america|latam|brasil)\b/i,
  brazil: /\b(brazil|mexico|argentina|chile|colombia|latin america|latam|brasil)\b/i,
  robotics: /\b(robot|robotics|humanoid|automation)\b/i,
  biotech: /\b(biotech|genome|crispr|pharma|vaccine)\b/i,
  climate: /\b(climate|warming|cop\d+|emissions|carbon)\b/i,
  energy: /\b(oil|gas|opec|renewable|solar|wind power|lng)\b/i,
  maritime: /\b(shipping|maritime|freight|supply chain|port|suez)\b/i,
  aviation: /\b(airline|aviation|aircraft|boeing|airbus|flight)\b/i,
  startup: /\b(startup|venture|unicorn|seed round|series [abc])\b/i,
  diplomacy: /\b(summit|treaty|un envoy|diplomat|ceasefire talks)\b/i,
  africa: /\b(africa|nigeria|kenya|south africa|ethiopia|ghana|sahel)\b/i,
  india: /\b(india|modi|delhi|mumbai|pakistan|bangladesh)\b/i,
  turkey: /\b(turkey|turkish|ankara|erdogan|istanbul)\b/i,
  iran: /\b(iran|tehran|persian|irgc)\b/i,
  russia: /\b(russia|russian|moscow|kremlin|putin)\b/i,
  nuclear: /\b(nuclear|reactor|uranium|iaea|warhead)\b/i,
  travel: /\b(travel|tourism|hotel|airport|visa)\b/i,
  religion: /\b(church|mosque|pope|faith|religious|synagogue)\b/i,
};

export function looseQueryMatch(item: RssItem, query: string): boolean {
  const hay = `${item.title} ${item.description}`.toLowerCase();
  const q = query.toLowerCase();
  if (hay.includes(q)) return true;
  const pattern = LOOSE_QUERY_PATTERNS[q];
  if (pattern?.test(hay)) return true;
  return false;
}

export function fallbackLaneHit(
  lane: TopicLane,
  rssItems: RssItem[],
  usedUrls: Set<string>,
): TopicDigestHit | null {
  const fresh = [...rssItems].sort((a, b) => b.publishedTs - a.publishedTs);
  const withImage = fresh.filter((item) => item.link && !usedUrls.has(item.link) && rssHasPreviewImage(item));
  const pool = withImage.length ? withImage : fresh;

  const categories = LANE_FEED_CATEGORIES[lane.id];
  if (categories?.length) {
    for (const item of pool) {
      if (!item.link || usedUrls.has(item.link)) continue;
      if (categories.includes(item.category)) {
        return hitFromRss(lane, item, 12, "low", true);
      }
    }
  }

  for (const item of pool) {
    if (!item.link || usedUrls.has(item.link)) continue;
    if (looseQueryMatch(item, lane.query)) {
      return hitFromRss(lane, item, 18, "low", true);
    }
  }

  for (const item of pool) {
    if (!item.link || usedUrls.has(item.link)) continue;
    return hitFromRss(lane, item, 5, "low", true);
  }

  return null;
}

async function pickLaneHits(
  lane: TopicLane,
  rssItems: RssItem[],
  articles: ArticleRecord[],
  indexedUrls: Set<string>,
  perLane: number,
  usedUrls: Set<string>,
  boostSourceKeys?: Set<string>,
): Promise<TopicDigestHit[]> {
  const rankedHeadlines = rankRssHeadlinesForQuery(rssItems, lane.query, indexedUrls, perLane * 8);
  const rankedIndexed = await rankIndexedArticlesForQuery(articles, lane.query, perLane * 8);

  const rssById = new Map(rssItems.map((r) => [r.id, r]));
  const articleById = new Map(articles.map((a) => [a.id, a]));

  const candidates: TopicDigestHit[] = [];

  for (const h of rankedHeadlines) {
    const item = rssById.get(h.id);
    if (!item) continue;
    const boost = boostSourceKeys?.has(item.sourceKey) ? 40 : 0;
    const imageBoost = rssHasPreviewImage(item) ? IMAGE_PICK_BOOST : 0;
    candidates.push({
      id: `${lane.id}:headline:${item.id}`,
      laneId: lane.id,
      laneLabel: lane.label,
      laneIcon: lane.icon,
      query: lane.query,
      score: h.score + boost + imageBoost,
      matchLabel: h.matchLabel,
      article: articleFromRss(item),
      sourceKind: "headline",
      fallback: false,
    });
  }

  for (const h of rankedIndexed) {
    const article = articleById.get(h.id);
    if (!article) continue;
    const boost = boostSourceKeys?.has(article.sourceKey) ? 40 : 0;
    const imageBoost = articleHasPreviewImage(article) ? IMAGE_PICK_BOOST : 0;
    candidates.push({
      id: `${lane.id}:indexed:${article.id}`,
      laneId: lane.id,
      laneLabel: lane.label,
      laneIcon: lane.icon,
      query: lane.query,
      score: h.score + boost + imageBoost,
      matchLabel: h.matchLabel,
      article,
      sourceKind: "indexed",
      fallback: false,
    });
  }

  candidates.sort((a, b) => b.score - a.score);

  const picked: TopicDigestHit[] = [];
  for (const c of candidates) {
    const url = c.article.url;
    if (!url || usedUrls.has(url)) continue;
    usedUrls.add(url);
    picked.push(c);
    if (picked.length >= perLane) break;
  }

  if (!picked.length) {
    const fb = fallbackLaneHit(lane, rssItems, usedUrls);
    if (fb) {
      usedUrls.add(fb.article.url);
      picked.push(fb);
    }
  }

  return picked;
}

export async function buildTopicsDigest(options: BuildTopicsDigestOptions = {}): Promise<TopicsDigest> {
  const perLane = options.perLane ?? TOPICS_PER_LANE;
  const lanes = options.lanes ?? TOPIC_LANES;
  const boostSourceKeys = options.boostSourceKeys;
  const rssLimit = options.lightMix ? 1200 : 3000;
  const articleLimit = options.lightMix ? 600 : 1200;

  const [rssItems, articles] = await Promise.all([
    getAllRssItems(rssLimit),
    getAllArticles(articleLimit),
  ]);
  const indexedUrls = new Set(articles.map((a) => a.url));

  const hitsByLane = new Map<string, TopicDigestHit[]>();
  const usedUrls = new Set<string>();

  for (let li = 0; li < lanes.length; li++) {
    const lane = lanes[li];
    const laneHits = await pickLaneHits(
      lane,
      rssItems,
      articles,
      indexedUrls,
      perLane,
      usedUrls,
      boostSourceKeys,
    );
    if (laneHits.length) hitsByLane.set(lane.id, laneHits);
    if (li % 2 === 1) await yieldToMain();
  }

  // Guarantee at least one card per lane when RSS data exists.
  for (const lane of lanes) {
    if ((hitsByLane.get(lane.id)?.length ?? 0) > 0) continue;
    const fb = fallbackLaneHit(lane, rssItems, usedUrls);
    if (fb) {
      usedUrls.add(fb.article.url);
      hitsByLane.set(lane.id, [fb]);
    }
  }

  let hits = lanes.flatMap((lane) => hitsByLane.get(lane.id) ?? []);

  const translated = await applyDisplayLanguageBatch(hits.map((h) => h.article));
  hits = hits.map((hit, i) => ({
    ...hit,
    article: translated[i] ?? hit.article,
  }));

  return {
    generatedAt: Date.now(),
    hits,
    stats: {
      totalLanes: lanes.length,
      lanesWithHits: hitsByLane.size,
      keywordHits: hits.filter((h) => !h.fallback).length,
      fallbackHits: hits.filter((h) => h.fallback).length,
      headlineHits: hits.filter((h) => h.sourceKind === "headline").length,
      indexedHits: hits.filter((h) => h.sourceKind === "indexed").length,
    },
  };
}
