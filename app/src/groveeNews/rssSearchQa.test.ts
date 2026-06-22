import { beforeEach, describe, expect, it, vi } from "vitest";
import { classifySearchIntents } from "../webSearch/intents";
import { buildUnifiedSearchPayload } from "../searchResults/mergeSearchHits";
import { isTopicsOverviewQuery } from "./headlineIntent";
import {
  extractNewsTopicTerms,
  isExplicitNewsTopicSearch,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import { NEWS_TOPIC_ACCEPTANCE_QUERIES } from "./newsTopicAcceptanceQueries";
import type { ArticleRecord, RssItem } from "./engine/types";
import {
  buildSearchTerms,
  expandQueryKeywords,
  isTopHitRelevant,
  rankArticlesForQuery,
  rankRssHeadlinesForQuery,
  rssItemToSearchArticle,
} from "./engine/search/relevance";
import { buildTopicsDigest } from "./engine/topics/buildTopicsDigest";
import { TOPIC_LANES } from "./engine/topics/topicLanes";
import { fetchTopicsBundle } from "./topicsAdapter";

// ─── Fixtures: synthetic RSS corpus (mirrors full-DB search) ─────────────────

const rss = (
  id: string,
  title: string,
  description: string,
  extras: Partial<RssItem> = {},
): RssItem => ({
  id,
  title,
  description,
  source: extras.source ?? "Wire",
  sourceKey: extras.sourceKey ?? "wire_en",
  category: extras.category ?? "world",
  link: `https://news.example.com/${id}`,
  image: "",
  published: new Date().toISOString(),
  publishedTs: Date.now() - Math.random() * 86_400_000,
  guid: id,
  ...extras,
});

const article = (
  id: string,
  title: string,
  summary: string,
  extras: Partial<ArticleRecord> = {},
): ArticleRecord => ({
  id,
  url: `https://news.example.com/${id}`,
  source: extras.source ?? "Indexed",
  sourceKey: extras.sourceKey ?? "wire_en",
  title,
  image: "",
  publishDate: new Date().toISOString(),
  publishedTs: Date.now(),
  articleText: summary,
  summary,
  keyFacts: extras.keyFacts ?? [],
  keywords: extras.keywords ?? [],
  entities: extras.entities ?? [],
  clusterId: id,
  confidence: "MEDIUM",
  fetchedAt: Date.now(),
  summarizedAt: Date.now(),
  ...extras,
});

/** Broad RSS pool — search ranks over all items (like production IndexedDB). */
const RSS_CORPUS: RssItem[] = [
  rss("pol-1", "Global politics shift as coalition talks stall", "Government leaders debate election reform and parliament votes.", { category: "world" }),
  rss("pol-2", "White House politics dominate Sunday talk shows", "Lawmakers discuss upcoming election and party strategy.", { category: "world" }),
  rss("pol-3", "European politics: migration policy divides bloc", "EU parliament faces vote on border rules.", { category: "world" }),
  rss("eco-1", "World economy faces slowdown amid trade tensions", "Economists warn of recession risk in major markets.", { category: "business" }),
  rss("eco-2", "Stock market rally lifts Nasdaq to record high", "Trading volume surges as investors buy tech shares.", { category: "business" }),
  rss("eco-3", "Israel economy grows despite regional conflict", "Central bank holds rates steady.", { category: "business", sourceKey: "il_globes" }),
  rss("sp-1", "NASA announces new Mars rover mission timeline", "Space agency plans launch for red planet exploration.", { category: "science" }),
  rss("sp-2", "SpaceX completes successful Starship test flight", "Private space industry reaches new orbit milestone.", { category: "science" }),
  rss("sp-3", "European Space Agency maps dark matter", "Science teams publish deep space survey results.", { category: "science" }),
  rss("ai-1", "OpenAI ships faster LLM for enterprise developers", "Artificial intelligence adoption accelerates in cloud.", { category: "technology" }),
  rss("ai-2", "Google DeepMind publishes robotics breakthrough", "Machine learning model improves warehouse automation.", { category: "technology" }),
  rss("war-1", "Ukraine war: frontline fighting intensifies", "Military conflict enters fourth year with heavy casualties.", { category: "world" }),
  rss("war-2", "Gaza war diplomacy stalls at UN", "Conflict mediation efforts continue in Middle East.", { category: "world", sourceKey: "il_ynet" }),
  rss("sport-1", "Champions League football final draws record crowd", "Sports fans celebrate championship weekend.", { category: "sport" }),
  rss("film-1", "Hollywood film festival opens with premiere", "Cinema industry celebrates new releases.", { category: "entertainment" }),
  rss("health-1", "Global health officials track new virus variant", "Medical teams expand hospital and medicine preparedness.", { category: "health" }),
  rss("il-1", "Israel politics: Knesset passes budget vote", "Netanyahu coalition survives key parliament test.", { sourceKey: "il_ynet", category: "israel" }),
  rss("il-2", "Tel Aviv tech startups raise venture funding", "Israeli founders announce Series B round.", { sourceKey: "il_calcalist", category: "israel" }),
  rss("cyber-1", "Major cyber breach hits financial sector", "Hackers exploit ransomware and security vulnerability.", { category: "technology" }),
  rss("cyber-2", "Cybersecurity firms warn of new security threat", "Security teams patch systems after global cyber attack.", { category: "technology" }),
  rss("climate-1", "Climate scientists warn of record heat wave", "Climate change drives environment agencies to issue emergency guidance.", { category: "science" }),
  rss("tech-1", "Technology startups raise record venture funding", "Tech founders announce new software platform launches.", { category: "technology" }),
  rss("weather-1", "Extreme weather floods coastal cities", "Storm systems bring heat wave and heavy rain worldwide.", { category: "science" }),
  rss("energy-1", "Renewable energy investment hits new high", "Solar and wind electric projects expand globally.", { category: "business" }),
  rss("ev-1", "Electric vehicle sales surge as Tesla expands", "Automakers invest in electric car and EV battery plants.", { category: "technology" }),
  rss("crypto-1", "Bitcoin crypto market rallies after ETF approval", "Ethereum blockchain trading volume climbs.", { category: "business" }),
  rss("bigtech-1", "Big tech earnings: Google Apple Microsoft report growth", "Amazon and Meta lead technology sector rally.", { category: "technology" }),
  rss("social-1", "Social media platforms face regulation push", "Facebook Meta TikTok Twitter debate content rules.", { category: "technology" }),
  rss("edu-1", "Education technology transforms university learning", "Schools adopt edtech platforms for remote education.", { category: "world" }),
  rss("reg-1", "Tech regulation bill advances in parliament", "Legal antitrust compliance rules target big technology firms.", { category: "world" }),
  rss("mil-1", "Military defense spending rises amid conflict", "Army Pentagon IDF announce defense procurement.", { category: "world" }),
  rss("game-1", "Gaming industry: PlayStation Xbox Nintendo showcase", "Video game publishers announce new releases.", { category: "entertainment" }),
  rss("ent-1", "Entertainment culture week: film music festival", "Hollywood movie premiere draws culture critics.", { category: "entertainment" }),
  rss("dis-1", "Breaking news: earthquake emergency disaster response", "Rescue teams rush after major disaster crash.", { category: "world" }),
  rss("agri-1", "Agriculture food tech improves crop yields", "Farm innovation boosts food production.", { category: "world" }),
  rss("start-il", "Israel startups close venture funding round", "Israeli founders launch AI startup accelerator.", { sourceKey: "il_calcalist", category: "israel" }),
];

const INDEXED_CORPUS: ArticleRecord[] = [
  article("idx-pol", "Parliament election reshapes national politics", "Voters head to polls amid fierce political debate.", {
    keywords: ["politics", "election"],
    entities: ["Parliament"],
  }),
  article("idx-eco", "Federal Reserve signals caution on economy", "Stock market volatility rises after rate decision.", {
    keywords: ["economy", "market", "stocks"],
  }),
  article("idx-space", "NASA telescope captures deep space imagery", "Science mission reveals distant galaxy formation.", {
    keywords: ["space", "nasa", "science"],
    entities: ["NASA"],
  }),
];

const rankFullCorpus = (query: string, limit = 40) => {
  const indexedUrls = new Set(INDEXED_CORPUS.map((a) => a.url));
  const headlineRanked = rankRssHeadlinesForQuery(RSS_CORPUS, query, indexedUrls, limit);
  const articleRanked = rankArticlesForQuery(INDEXED_CORPUS, query, limit);
  const byId = new Map([...RSS_CORPUS.map((r) => [r.id, r.title]), ...INDEXED_CORPUS.map((a) => [a.id, a.title])]);
  return {
    headlineRanked,
    articleRanked,
    titles: [...headlineRanked, ...articleRanked].map((h) => byId.get(h.id) ?? h.id),
    total: headlineRanked.length + articleRanked.length,
  };
};

const titleHaystack = (titles: string[]) => titles.join(" ").toLowerCase();

// ─── Search matrix: query → expected behaviour ───────────────────────────────

type SearchCase = {
  id: string;
  query: string;
  minHits: number;
  engineIncludes?: string;
  titleMatch?: RegExp;
  titleAvoid?: RegExp;
  wordCount?: number;
};

const SEARCH_MATRIX: SearchCase[] = [
  { id: "RSS-S01", query: "politics", minHits: 3, titleMatch: /politic|election|parliament|government/i },
  { id: "RSS-S02", query: "פוליטיקה", minHits: 2, engineIncludes: "politics", titleMatch: /politic|election|knesset|netanyahu/i },
  { id: "RSS-S03", query: "economy", minHits: 2, titleMatch: /economy|market|stock|recession/i },
  { id: "RSS-S04", query: "כלכלה", minHits: 1, engineIncludes: "economy", titleMatch: /economy|market|bank/i },
  { id: "RSS-S05", query: "space", minHits: 3, titleMatch: /space|nasa|mars|starship|galaxy/i },
  { id: "RSS-S06", query: "חלל", minHits: 2, engineIncludes: "space", titleMatch: /space|nasa|science/i },
  { id: "RSS-S07", query: "artificial intelligence", minHits: 1, titleMatch: /ai|artificial|learning|openai|robot/i },
  { id: "RSS-S08", query: "stock market", minHits: 1, wordCount: 2, titleMatch: /stock|market|nasdaq|trading/i },
  { id: "RSS-S09", query: "ukraine war", minHits: 1, wordCount: 2, titleMatch: /ukraine|war|conflict|military/i },
  { id: "RSS-S10", query: "nasa space", minHits: 1, wordCount: 2, titleMatch: /nasa|space/i },
  { id: "RSS-S11", query: "israel politics", minHits: 1, wordCount: 2, titleMatch: /israel|knesset|netanyahu|politic/i },
  { id: "RSS-S12", query: "cyber security", minHits: 1, wordCount: 2, titleMatch: /cyber|hack|breach|ransom/i },
  { id: "RSS-S13", query: "climate change", minHits: 1, wordCount: 2, titleMatch: /climate|heat|environment/i },
  { id: "RSS-S14", query: "hollywood film", minHits: 1, wordCount: 2, titleMatch: /hollywood|film|cinema/i, titleAvoid: /politic|nasa|war/i },
  { id: "RSS-S15", query: "earthquake", minHits: 1, titleMatch: /earthquake|disaster|emergency|seismic/i },
  { id: "RSS-S16", query: "flood", minHits: 1, titleMatch: /flood|inundation|storm|weather/i },
  { id: "RSS-S17", query: "רעידות אדמה אחרונות", minHits: 1, engineIncludes: "earthquake", titleMatch: /earthquake|disaster|emergency/i },
  { id: "RSS-S18", query: "הצפה", minHits: 1, engineIncludes: "flood", titleMatch: /flood|weather|storm|rain/i },
];

const TOPICS_OVERVIEW_QUERIES = [
  "מה קורה בעולם?",
  "מה חדש בעולם?",
  "what's happening in the world",
  "world news headlines",
];

const KEYWORD_NEWS_QUERIES = [
  "פוליטיקה",
  "כלכלה",
  "חלל",
  "חפש חדשות בנושא בינה מלאכותית",
  "חדשות כלכלה",
];

// ─── DB mock for Topics digest ───────────────────────────────────────────────

const getAllRssItemsMock = vi.fn();
const getAllArticlesMock = vi.fn();

vi.mock("./engine/storage/db", () => ({
  getAllRssItems: (...args: unknown[]) => getAllRssItemsMock(...args),
  getAllArticles: (...args: unknown[]) => getAllArticlesMock(...args),
}));

vi.mock("./engine/display/liveFeedDisplay", () => ({
  applyDisplayLanguageBatch: (articles: ArticleRecord[]) => Promise.resolve(articles),
}));

vi.mock("./engine/search/flexIndex", () => ({
  rankIndexedArticlesForQuery: async (articles: ArticleRecord[], query: string, limit: number) =>
    rankArticlesForQuery(articles, query, limit),
}));

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("RSS search QA — multi-word query parsing", () => {
  it.each([
    ["stock market", 2],
    ["ukraine war", 2],
    ["nasa space", 2],
    ["israel politics", 2],
    ["artificial intelligence", 2],
  ])('"%s" expands to %i required keyword(s)', (query, count) => {
    expect(expandQueryKeywords(query).length).toBe(count);
    expect(buildSearchTerms(query).length).toBeGreaterThan(0);
  });

  it("Hebrew queries normalize to English engine terms", () => {
    expect(normalizeNewsEngineQuery("פוליטיקה")).toBe("politics");
    expect(normalizeNewsEngineQuery("כלכלה")).toBe("economy");
    expect(normalizeNewsEngineQuery("חלל")).toBe("space");
    expect(buildSearchTerms(normalizeNewsEngineQuery("פוליטיקה")).length).toBeGreaterThan(0);
  });

  it("two-word queries require both terms in top hit", () => {
    const { headlineRanked } = rankFullCorpus("ukraine war");
    expect(headlineRanked.length).toBeGreaterThan(0);
    const top = RSS_CORPUS.find((r) => r.id === headlineRanked[0].id)!;
    expect(isTopHitRelevant(rssItemToSearchArticle(top), "ukraine war")).toBe(true);
  });
});

describe("RSS search QA — full-corpus ranking matrix", () => {
  it.each(SEARCH_MATRIX.map((c) => [c.id, c]))("%s: %s", (_id, c) => {
    const engineQ = normalizeNewsEngineQuery(c.query);
    if (c.engineIncludes) {
      expect(engineQ.toLowerCase()).toContain(c.engineIncludes);
    }
    if (c.wordCount) {
      expect(expandQueryKeywords(engineQ || c.query).length).toBeGreaterThanOrEqual(c.wordCount);
    }

    const { total, titles } = rankFullCorpus(engineQ || c.query);
    expect(total, `too few hits for «${c.query}» → engine «${engineQ}»`).toBeGreaterThanOrEqual(c.minHits);

    const hay = titleHaystack(titles);
    if (c.titleMatch) {
      expect(hay).toMatch(c.titleMatch);
    }
    if (c.titleAvoid) {
      expect(hay).not.toMatch(c.titleAvoid);
    }
  });
});

describe("RSS search QA — Hebrew topic acceptance queries", () => {
  it.each(NEWS_TOPIC_ACCEPTANCE_QUERIES.map((q) => [q.id, q]))("%s normalizes and ranks", (_id, spec) => {
    expect(classifySearchIntents(spec.query)).toEqual(expect.arrayContaining(spec.expectIntents));
    expect(normalizeNewsEngineQuery(spec.query)).toBe(spec.expectEngineQuery);

    const { titles, total } = rankFullCorpus(spec.expectEngineQuery);
    expect(total).toBeGreaterThan(0);

    const hay = titleHaystack(titles);
    const matched = spec.expectTitleKeywords.some((kw) => hay.includes(kw.toLowerCase()));
    expect(matched, `expected one of [${spec.expectTitleKeywords.join(", ")}] in results for ${spec.id}`).toBe(true);
  });
});

describe("RSS search QA — Topics vs keyword routing", () => {
  it.each(TOPICS_OVERVIEW_QUERIES.map((q) => [q]))("overview «%s» → Topics mode", (q) => {
    expect(isTopicsOverviewQuery(q)).toBe(true);
    expect(isExplicitNewsTopicSearch(q)).toBe(false);
  });

  it.each(KEYWORD_NEWS_QUERIES.map((q) => [q]))("keyword «%s» → search mode (not Topics overview)", (q) => {
    expect(isTopicsOverviewQuery(q)).toBe(false);
  });

  it("explicit topic phrase is not world digest", () => {
    const q = "חפש חדשות בנושא מדע וחלל";
    expect(isExplicitNewsTopicSearch(q)).toBe(true);
    expect(isTopicsOverviewQuery(q)).toBe(false);
    expect(extractNewsTopicTerms(q).length).toBeGreaterThan(0);
  });
});

describe("RSS search QA — Topics digest mixed lanes", () => {
  beforeEach(() => {
    getAllRssItemsMock.mockResolvedValue(RSS_CORPUS);
    getAllArticlesMock.mockResolvedValue(INDEXED_CORPUS);
  });

  it("returns hits from many distinct topic lanes", async () => {
    const digest = await buildTopicsDigest({ perLane: 1, lanes: TOPIC_LANES });
    const laneIds = new Set(digest.hits.map((h) => h.laneId));
    const urls = digest.hits.map((h) => h.article.url);

    expect(digest.stats.totalLanes).toBe(TOPIC_LANES.length);
    expect(digest.stats.lanesWithHits).toBeGreaterThanOrEqual(12);
    expect(laneIds.size).toBeGreaterThanOrEqual(12);
    expect(new Set(urls).size).toBe(urls.length);
    expect(digest.hits.length).toBeGreaterThanOrEqual(20);
  });

  it("each lane hit carries lane metadata and search query", async () => {
    const sampleLanes = TOPIC_LANES.filter((l) =>
      ["politics", "market", "space", "ai", "war", "israel"].includes(l.id),
    );
    const digest = await buildTopicsDigest({ perLane: 2, lanes: sampleLanes });

    for (const hit of digest.hits) {
      expect(hit.laneId).toBeTruthy();
      expect(hit.laneLabel).toBeTruthy();
      expect(hit.laneIcon).toBeTruthy();
      expect(hit.query).toBeTruthy();
      expect(hit.article.title.length).toBeGreaterThan(5);
      expect(hit.article.url).toMatch(/^https?:\/\//);
    }

    const laneIds = new Set(digest.hits.map((h) => h.laneId));
    expect(laneIds.size).toBe(sampleLanes.length);
  });

  it("politics / economy / space lanes return domain-relevant headlines", async () => {
    const lanes = TOPIC_LANES.filter((l) => ["politics", "market", "space"].includes(l.id));
    const digest = await buildTopicsDigest({ perLane: 2, lanes });

    const byLane = new Map<string, string[]>();
    for (const h of digest.hits) {
      const list = byLane.get(h.laneId) ?? [];
      list.push(h.article.title.toLowerCase());
      byLane.set(h.laneId, list);
    }

    expect(byLane.get("politics")?.join(" ")).toMatch(/politic|election|parliament|government|knesset/i);
    expect(byLane.get("market")?.join(" ")).toMatch(/market|stock|economy|trading|nasdaq/i);
    expect(byLane.get("space")?.join(" ")).toMatch(/space|nasa|mars|science|galaxy/i);
  });

  it("fetchTopicsBundle exposes cards with lane labels for UI mix", async () => {
    const bundle = await fetchTopicsBundle();
    expect(bundle.cards.length).toBeGreaterThanOrEqual(12);
    expect(bundle.stats.lanesWithHits).toBeGreaterThanOrEqual(8);

    const withLane = bundle.cards.filter((c) => c.laneId && c.laneLabel);
    expect(withLane.length).toBe(bundle.cards.length);

    const uniqueLanes = new Set(bundle.cards.map((c) => c.laneId));
    expect(uniqueLanes.size).toBeGreaterThanOrEqual(8);
  });
});

describe("RSS search QA — SERP payload from Topics overview", () => {
  beforeEach(() => {
    getAllRssItemsMock.mockResolvedValue(RSS_CORPUS);
    getAllArticlesMock.mockResolvedValue(INDEXED_CORPUS);
  });

  it("world overview produces multi-lane RSS facet in unified payload", async () => {
    const bundle = await fetchTopicsBundle();
    const sources = [
      {
        provider: "grovee-news" as const,
        label: `חדשות (Topics · ${bundle.stats.lanesWithHits} נושאים)`,
        ok: true,
        text: "ANSWER",
        newsCards: bundle.cards,
        latencyMs: 10,
      },
    ];
    const payload = buildUnifiedSearchPayload("מה קורה בעולם?", sources);

    expect(payload.preferRssFilter).toBe(true);
    expect(payload.facets.rss).toBeGreaterThanOrEqual(12);
    expect(payload.hits.filter((h) => h.kind === "rss").length).toBeGreaterThanOrEqual(12);

    const laneLabels = new Set(bundle.cards.map((c) => c.laneLabel));
    expect(laneLabels.size).toBeGreaterThanOrEqual(8);
  });
});

describe("RSS search QA — result volume guards", () => {
  it("single-topic queries return more than 1 hit when corpus is broad", () => {
    for (const topic of ["politics", "economy", "space", "tech", "war"]) {
      const { total } = rankFullCorpus(topic, 40);
      expect(total, `expected multiple hits for «${topic}»`).toBeGreaterThanOrEqual(2);
    }
  });

  it("rankRssHeadlinesForQuery scans full RSS pool (not single feed)", () => {
    const hits = rankRssHeadlinesForQuery(RSS_CORPUS, "politics", new Set(), 40);
    expect(hits.length).toBeGreaterThanOrEqual(3);
    const sourceKeys = new Set(
      hits.map((h) => RSS_CORPUS.find((r) => r.id === h.id)?.sourceKey).filter(Boolean),
    );
    expect(sourceKeys.size).toBeGreaterThanOrEqual(1);
  });
});
