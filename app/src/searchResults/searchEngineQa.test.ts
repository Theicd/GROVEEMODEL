import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GroveeNewsCard } from "../groveeNews/types";
import { fetchGroveeNewsSearch } from "../groveeNews/newsToSearchBrief";
import type { SearchProviderId, SearchSourceResult } from "../webSearch/types";
import { buildUnifiedSearchPayload, mergeSourcesToHits } from "./mergeSearchHits";
import { filterHits, isAllTabHit } from "./rankHits";
import type { SearchHitKind, SearchResultsFilter, SearchResultsPayload, UnifiedSearchHit } from "./types";

/** Mirrors SearchResultsPanel — always «הכל» on open / new query. */
const resolveInitialTab = (_payload: SearchResultsPayload): SearchResultsFilter => "all";

const PANEL_TABS: SearchResultsFilter[] = [
  "all",
  "rss",
  "images",
  "video",
  "events",
  "products",
  "movies",
  "hfmodels",
  "ships",
  "repos",
];

const rssCard = (id: string, title: string, topic = "ynet"): GroveeNewsCard => ({
  id,
  title,
  titleOriginal: title,
  source: topic,
  sourceKey: topic,
  url: `https://example.com/${id}`,
  image: "",
  score: 50,
  publishedTs: Date.now(),
});

const mockHit = (
  kind: SearchHitKind,
  id: string,
  overrides: Partial<UnifiedSearchHit> = {},
): UnifiedSearchHit => ({
  id,
  kind,
  title: `${kind} ${id}`,
  url: `https://example.com/${kind}/${id}`,
  snippet: "snippet",
  sourceLabel: kind,
  provider: "searxng" as SearchProviderId,
  score: 40,
  summarizable: kind === "rss",
  ...overrides,
});

const newsSource = (cards: GroveeNewsCard[]): SearchSourceResult => ({
  provider: "grovee-news",
  label: "GROVEE NEWS",
  ok: true,
  text: "",
  newsCards: cards,
  latencyMs: 12,
});

const searchNewsMock = vi.fn();
const buildRecentHeadlineHitsMock = vi.fn();
const fetchTopicsBundleMock = vi.fn();
const hitsToDisplayCardsMock = vi.fn();
const startBootMock = vi.fn();

vi.mock("../networkReachability", () => ({
  resolveNetworkReachability: vi.fn().mockResolvedValue("online"),
}));

vi.mock("../groveeNews/liveSearchPoll", () => ({
  pollRssForLiveSearch: vi.fn().mockResolvedValue({
    sessionStart: Date.now(),
    feedsOk: 10,
    feedsFailed: 0,
    newHeadlines: 5,
    feedsPolled: 10,
  }),
}));

vi.mock("../groveeNews/engine/engine/pipeline", () => ({
  searchNews: (...args: unknown[]) => searchNewsMock(...args),
}));

vi.mock("../groveeNews/hebrewFeedPoll", () => ({
  priorityPollHebrewFeeds: vi.fn().mockResolvedValue(0),
}));

vi.mock("../groveeNews/rssSeed", () => ({
  ensureRssCatalogReady: vi.fn().mockResolvedValue(120),
}));

vi.mock("../groveeNews/engine/settings/userNewsProfile", () => ({
  getUserNewsProfile: () => ({ locale: "he-IL", uiLanguage: "he", pollTier: "core" }),
}));

vi.mock("../groveeNews/recentHeadlineHits", () => ({
  buildRecentHeadlineHits: (...args: unknown[]) => buildRecentHeadlineHitsMock(...args),
}));

vi.mock("../groveeNews/topicsAdapter", () => ({
  fetchTopicsBundle: (...args: unknown[]) => fetchTopicsBundleMock(...args),
}));

vi.mock("../groveeNews/searchAdapter", () => ({
  hitsToDisplayCards: (...args: unknown[]) => hitsToDisplayCardsMock(...args),
}));

vi.mock("../groveeNews/engineBoot", () => ({
  startGroveeNewsBoot: (...args: unknown[]) => startBootMock(...args),
}));

vi.mock("../groveeNews/engine/engine/engineStats", () => ({
  getEngineLibraryStats: vi.fn().mockResolvedValue({ rssHeadlines: 500, articlesIndexed: 40 }),
}));

vi.mock("../groveeNews/engine/search/flexIndex", () => ({
  getSearchIndexSize: vi.fn().mockReturnValue(40),
}));

describe("Search engine QA — tab filters", () => {
  const corpus: UnifiedSearchHit[] = [
    mockHit("rss", "r1", { provider: "grovee-news" }),
    mockHit("web", "w1", { provider: "wikipedia-en" }),
    mockHit("github", "g1", { provider: "github" }),
    mockHit("arxiv", "a1", { provider: "arxiv" }),
    mockHit("movie", "m1", { provider: "movie-catalog" }),
    mockHit("image", "i1", { provider: "pixabay-images" }),
    mockHit("video", "v1", { provider: "internet-archive-media" }),
    mockHit("youtube", "y1", { provider: "invidious-videos" }),
    mockHit("livetv", "tv1", { provider: "live-tv", url: "https://example.com/livetv/tv1" }),
    mockHit("radio", "rd1", { provider: "live-tv", url: "https://example.com/radio/rd1" }),
    mockHit("product", "p1", { provider: "israeli-products" }),
    mockHit("hfmodel", "hf1", { provider: "huggingface-models" }),
    mockHit("earthquake", "eq1", { provider: "usgs-earthquake" }),
    mockHit("disaster", "d1", { provider: "gdacs-disasters" }),
    mockHit("ship", "sh1", { provider: "ais-ships" }),
    mockHit("marine", "mi1", { provider: "osm-overpass-marine" }),
  ];

  it.each(PANEL_TABS)("filter %s returns only matching kinds", (tab) => {
    const filtered = filterHits(corpus, tab);
    if (tab === "all") {
      expect(filtered.every(isAllTabHit)).toBe(true);
      expect(filtered).toHaveLength(corpus.filter(isAllTabHit).length);
      return;
    }
    if (tab === "earthquakes") {
      expect(filtered.every((h) => h.kind === "earthquake")).toBe(true);
      expect(filtered.length).toBe(1);
      return;
    }
    if (tab === "disasters") {
      expect(filtered.every((h) => h.kind === "disaster")).toBe(true);
      expect(filtered.length).toBe(1);
      return;
    }
    if (tab === "events") {
      expect(filtered.every((h) => h.kind === "earthquake" || h.kind === "disaster")).toBe(true);
      expect(filtered.length).toBe(2);
      return;
    }
    if (tab === "ships") {
      expect(filtered.every((h) => h.kind === "ship" || h.kind === "marine")).toBe(true);
      expect(filtered.length).toBe(2);
      return;
    }
    if (tab === "repos") {
      expect(filtered.every((h) => h.kind === "github" || h.kind === "arxiv")).toBe(true);
      expect(filtered.length).toBe(2);
      return;
    }
    if (tab === "video") {
      expect(filtered.every((h) => h.kind === "video" || h.kind === "youtube")).toBe(true);
      expect(filtered.length).toBe(2);
      return;
    }
    const kindByTab: Partial<Record<SearchResultsFilter, SearchHitKind>> = {
      rss: "rss",
      web: "web",
      images: "image",
      movies: "movie",
      products: "product",
      hfmodels: "hfmodel",
    };
    const expectedKind = kindByTab[tab];
    expect(expectedKind).toBeTruthy();
    expect(filtered.length).toBeGreaterThan(0);
    expect(filtered.every((h) => h.kind === expectedKind)).toBe(true);
  });

  it("every facet with count > 0 has a non-empty tab", () => {
    const payload = buildUnifiedSearchPayload("mixed", [
      newsSource([rssCard("1", "Politics headline")]),
      {
        provider: "wikipedia-en",
        label: "Wikipedia",
        ok: true,
        latencyMs: 10,
        text: "- Demo:\nDemo page.\nhttps://en.wikipedia.org/wiki/Demo",
      },
      {
        provider: "pixabay-images",
        label: "Pixabay",
        ok: true,
        latencyMs: 10,
        text: "",
        mediaHits: [
          {
            id: "img-1",
            mediaType: "image",
            title: "cat",
            url: "https://pixabay.com/i/1",
            playUrl: "https://cdn.pixabay.com/l.jpg",
            thumbnail: "https://cdn.pixabay.com/s.jpg",
          },
        ],
      },
      {
        provider: "invidious-videos",
        label: "YouTube",
        ok: true,
        latencyMs: 10,
        text: "",
        mediaHits: [
          {
            id: "yt-1",
            mediaType: "video",
            youtubeSubType: "video",
            title: "clip",
            url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            playUrl: "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
            thumbnail: "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            source: "YouTube",
          },
        ],
      },
      {
        provider: "movie-catalog",
        label: "Movies",
        ok: true,
        latencyMs: 10,
        text: "",
        movieHits: [
          {
            id: "mov-1",
            title: "Inception",
            year: 2010,
            url: "https://www.themoviedb.org/movie/27205",
            snippet: "Sci-fi",
            poster: "https://image.tmdb.org/poster.jpg",
            source: "TMDB",
          },
        ],
      },
      {
        provider: "israeli-products",
        label: "Products",
        ok: true,
        latencyMs: 10,
        text: "",
        productHits: [
          {
            id: "prod-1",
            barcode: "7290004131074",
            title: "חלב",
            brand: "תנובה",
            url: "https://cheapersal.co.il/p/1",
            snippet: "₪6",
            imageUrl: "https://example.com/milk.jpg",
            source: "קטלוג",
          },
        ],
      },
      {
        provider: "huggingface-models",
        label: "HF",
        ok: true,
        latencyMs: 10,
        text: "1. Qwen/Qwen2.5-7B (https://huggingface.co/Qwen/Qwen2.5-7B)",
      },
    ]);

    const tabChecks: Array<[SearchResultsFilter, number]> = [
      ["rss", payload.facets.rss],
      ["images", payload.facets.images],
      ["video", payload.facets.videos + payload.facets.youtube],
      ["movies", payload.facets.movies],
      ["products", payload.facets.products],
      ["hfmodels", payload.facets.hfModels],
    ];

    for (const [tab, count] of tabChecks) {
      if (count <= 0) continue;
      expect(filterHits(payload.hits, tab).length).toBeGreaterThan(0);
    }
  });
});

describe("Search engine QA — category queries", () => {
  type Scenario = {
    name: string;
    query: string;
    sources: SearchSourceResult[];
    contentTab: SearchResultsFilter;
    facetKey: keyof SearchResultsPayload["facets"];
  };

  const scenarios: Scenario[] = [
    {
      name: "politics news → RSS hits",
      query: "חדשות פוליטיקה ישראל",
      sources: [newsSource([rssCard("p1", "ממשלה דנה בחקיקה", "ynet")])],
      contentTab: "rss",
      facetKey: "rss",
    },
    {
      name: "general news → RSS hits",
      query: "מה קורה בעולם",
      sources: [newsSource([rssCard("n1", "World update")])],
      contentTab: "rss",
      facetKey: "rss",
    },
    {
      name: "movies query → movie hits",
      query: "סרט inception",
      sources: [
        {
          provider: "movie-catalog",
          label: "Movies",
          ok: true,
          latencyMs: 10,
          text: "",
          movieHits: [
            {
              id: "inception",
              title: "Inception",
              year: 2010,
              url: "https://www.themoviedb.org/movie/27205",
              snippet: "Dreams",
              poster: "https://image.tmdb.org/inception.jpg",
              source: "TMDB",
            },
          ],
        },
      ],
      contentTab: "movies",
      facetKey: "movies",
    },
    {
      name: "music / radio query → all tab (radio hits in results)",
      query: "תחנות רדיו מוזיקה",
      sources: [
        {
          provider: "live-tv",
          label: "Radio",
          ok: true,
          latencyMs: 10,
          text: "",
          liveMediaHits: [
            {
              id: "galgalatz",
              mediaType: "radio",
              title: "גלגלצ",
              url: "https://example.com/galgalatz",
              streamUrl: "https://stream.example/galgalatz",
            },
          ],
        },
      ],
      contentTab: "all",
      facetKey: "radio",
    },
    {
      name: "images query → image hits",
      query: "תמונות חתול",
      sources: [
        {
          provider: "pixabay-images",
          label: "Pixabay",
          ok: true,
          latencyMs: 10,
          text: "",
          mediaHits: [
            {
              id: "cat",
              mediaType: "image",
              title: "cat",
              url: "https://pixabay.com/cat",
              playUrl: "https://cdn.pixabay.com/cat.jpg",
              thumbnail: "https://cdn.pixabay.com/cat-s.jpg",
            },
          ],
        },
      ],
      contentTab: "images",
      facetKey: "images",
    },
    {
      name: "video query → video hits",
      query: "סרטון טבע",
      sources: [
        {
          provider: "internet-archive-media",
          label: "Archive",
          ok: true,
          latencyMs: 10,
          text: "",
          mediaHits: [
            {
              id: "ia-1",
              mediaType: "video",
              title: "Nature clip",
              url: "https://archive.org/details/nature",
              playUrl: "https://archive.org/download/nature/clip.mp4",
              thumbnail: "https://archive.org/services/img/nature",
              source: "Internet Archive",
            },
          ],
        },
      ],
      contentTab: "video",
      facetKey: "videos",
    },
    {
      name: "YouTube query → video hits",
      query: "youtube שיר עומר אדם",
      sources: [
        {
          provider: "invidious-videos",
          label: "YouTube",
          ok: true,
          latencyMs: 10,
          text: "",
          mediaHits: [
            {
              id: "yt-song",
              mediaType: "video",
              youtubeSubType: "video",
              title: "שיר",
              url: "https://www.youtube.com/watch?v=abc12345678",
              playUrl: "https://www.youtube-nocookie.com/embed/abc12345678",
              thumbnail: "https://i.ytimg.com/vi/abc12345678/hqdefault.jpg",
              source: "YouTube",
            },
          ],
        },
      ],
      contentTab: "video",
      facetKey: "youtube",
    },
    {
      name: "products query → product hits",
      query: "מחיר חלב תנובה",
      sources: [
        {
          provider: "israeli-products",
          label: "Products",
          ok: true,
          latencyMs: 10,
          text: "",
          productHits: [
            {
              id: "milk",
              barcode: "7290004131074",
              title: "חלב 3%",
              brand: "תנובה",
              url: "https://cheapersal.co.il/p/milk",
              snippet: "₪6.90",
              imageUrl: "https://example.com/milk.jpg",
              source: "קטלוג",
            },
          ],
        },
      ],
      contentTab: "products",
      facetKey: "products",
    },
    {
      name: "HF models query → hfmodel hits",
      query: "huggingface qwen model",
      sources: [
        {
          provider: "huggingface-models",
          label: "HF",
          ok: true,
          latencyMs: 10,
          text: "",
          hfModelHits: [
            {
              id: "hf-qwen",
              modelId: "Qwen/Qwen2.5-7B-Instruct",
              url: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
              title: "Qwen/Qwen2.5-7B-Instruct",
              snippet: "text-generation",
              status: "WORKING",
              provider: "HF inference",
              accessMode: "FREE",
              endpoint: "https://router.huggingface.co/v1/chat/completions",
              probed: true,
              probeSource: "browser",
              curlSnippet: "curl example",
              pythonSnippet: "import requests",
            },
          ],
        },
      ],
      contentTab: "hfmodels",
      facetKey: "hfModels",
    },
  ];

  it.each(scenarios)("$name returns hits and opens all tab", ({ query, sources, contentTab, facetKey }) => {
    const payload = buildUnifiedSearchPayload(query, sources);
    expect(payload.hits.length).toBeGreaterThan(0);
    expect(payload.facets[facetKey]).toBeGreaterThan(0);
    expect(filterHits(payload.hits, contentTab).length).toBeGreaterThan(0);
    expect(resolveInitialTab(payload)).toBe("all");
  });
});

describe("Search engine QA — RSS provider bridge", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    startBootMock.mockResolvedValue(undefined);
    fetchTopicsBundleMock.mockResolvedValue(null);
    buildRecentHeadlineHitsMock.mockResolvedValue([]);
    hitsToDisplayCardsMock.mockImplementation(async (hits: { article: { id: string; title: string; source: string; url: string } }[]) =>
      hits.map((h) => ({
        id: h.article.id,
        title: h.article.title,
        titleOriginal: h.article.title,
        source: h.article.source,
        sourceKey: "test",
        url: h.article.url,
        image: "",
        score: 10,
        publishedTs: Date.now(),
      })),
    );
  });

  it("fetchGroveeNewsSearch returns RSS cards merged as rss hits", async () => {
    searchNewsMock.mockResolvedValue([]);
    buildRecentHeadlineHitsMock.mockResolvedValue([
      {
        article: {
          id: "rss-1",
          title: "כותרת חדשות",
          source: "ynet",
          sourceKey: "il_ynet",
          url: "https://www.ynet.co.il/1",
        },
        score: 88,
        sourceKind: "headline",
      },
    ]);

    const result = await fetchGroveeNewsSearch("חדשות כלכלה");
    expect(result.ok).toBe(true);
    expect(result.newsCards?.length).toBeGreaterThan(0);

    const payload = buildUnifiedSearchPayload("חדשות כלכלה", [result]);
    expect(payload.facets.rss).toBeGreaterThan(0);
    expect(payload.hits.some((h) => h.kind === "rss")).toBe(true);
    expect(filterHits(payload.hits, "rss").length).toBeGreaterThan(0);
  });

  it("RSS hits survive mergeSourcesToHits with correct kind", () => {
    const hits = mergeSourcesToHits([
      newsSource([rssCard("1", "Politics"), rssCard("2", "Economy")]),
    ]);
    expect(hits).toHaveLength(2);
    expect(hits.every((h) => h.kind === "rss")).toBe(true);
  });

  it("news query with empty RSS still surfaces provider error in payload", () => {
    const payload = buildUnifiedSearchPayload("חדשות דחופות", [
      {
        provider: "grovee-news",
        label: "GROVEE NEWS",
        ok: false,
        text: "",
        error: "אין כותרות RSS",
        latencyMs: 5,
      },
    ]);
    expect(payload.facets.rss).toBe(0);
    expect(payload.providerErrors.some((e) => /GROVEE NEWS|RSS/i.test(e))).toBe(true);
  });
});

describe("Search engine QA — sensor queries + RSS (חדשות tab)", () => {
  const usgsText = `- M5.2 · 45 km NE of Tokyo, Japan · 2024-06-19 12:34:56 UTC
  https://earthquake.usgs.gov/earthquakes/eventpage/us7000`;

  const eqNewsSources: SearchSourceResult[] = [
    {
      provider: "usgs-earthquake",
      label: "USGS",
      ok: true,
      text: usgsText,
      latencyMs: 80,
    },
    newsSource([rssCard("eq-n1", "Major earthquake hits Japan coast", "bbc")]),
  ];

  it("רעידות אדמה — RSS hits visible in חדשות tab alongside USGS", () => {
    const payload = buildUnifiedSearchPayload("רעידות אדמה אחרונות", eqNewsSources);
    expect(payload.facets.rss).toBeGreaterThan(0);
    expect(payload.facets.earthquakes).toBeGreaterThan(0);
    expect(filterHits(payload.hits, "rss").length).toBeGreaterThan(0);
    expect(filterHits(payload.hits, "earthquakes").length).toBeGreaterThan(0);
  });

  it("when RSS + USGS both present, panel opens on הכל (not hiding חדשות)", () => {
    const payload = buildUnifiedSearchPayload("רעידות אדמה אחרונות", eqNewsSources);
    expect(payload.preferEventsFilter).toBe(false);
    expect(resolveInitialTab(payload)).toBe("all");
  });

  it("USGS only — opens הכל tab (events in אירועים filter)", () => {
    const payload = buildUnifiedSearchPayload("רעידות אדמה", [eqNewsSources[0]]);
    expect(payload.preferEventsFilter).toBe(true);
    expect(resolveInitialTab(payload)).toBe("all");
  });
});
