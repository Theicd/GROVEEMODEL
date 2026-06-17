import { beforeEach, describe, expect, it, vi } from "vitest";
import { classifySearchIntents } from "../webSearch/intents";
import { fetchGroveeNewsSearch } from "./newsToSearchBrief";
import { isTopicsOverviewQuery } from "./headlineIntent";
import { normalizeNewsEngineQuery } from "./newsQueryNormalize";
import { clearNewsPanelPayload, getNewsPanelPayload } from "./newsPanelStore";

const searchNewsMock = vi.fn();
const buildRecentHeadlineHitsMock = vi.fn();
const fetchTopicsBundleMock = vi.fn();
const hitsToDisplayCardsMock = vi.fn();
const startBootMock = vi.fn();
const getEngineLibraryStatsMock = vi.fn();
const getSearchIndexSizeMock = vi.fn();

vi.mock("./engine/engine/pipeline", () => ({
  searchNews: (...args: unknown[]) => searchNewsMock(...args),
}));

vi.mock("./recentHeadlineHits", () => ({
  buildRecentHeadlineHits: (...args: unknown[]) => buildRecentHeadlineHitsMock(...args),
}));

vi.mock("./topicsAdapter", () => ({
  fetchTopicsBundle: (...args: unknown[]) => fetchTopicsBundleMock(...args),
}));

vi.mock("./searchAdapter", () => ({
  hitsToDisplayCards: (...args: unknown[]) => hitsToDisplayCardsMock(...args),
}));

vi.mock("./engineBoot", () => ({
  startGroveeNewsBoot: (...args: unknown[]) => startBootMock(...args),
}));

vi.mock("./engine/engine/engineStats", () => ({
  getEngineLibraryStats: (...args: unknown[]) => getEngineLibraryStatsMock(...args),
}));

vi.mock("./engine/search/flexIndex", () => ({
  getSearchIndexSize: (...args: unknown[]) => getSearchIndexSizeMock(...args),
}));

describe("news chat bridge routing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearNewsPanelPayload();
    startBootMock.mockResolvedValue(undefined);
    getEngineLibraryStatsMock.mockResolvedValue({ rssHeadlines: 3201, articlesIndexed: 40 });
    getSearchIndexSizeMock.mockReturnValue(40);
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

  it("classifies Hebrew news queries with news intent", () => {
    expect(classifySearchIntents("חפש חדשות על חלל")).toContain("news");
    expect(classifySearchIntents("מה קורה בעולם?")).toContain("news");
    expect(classifySearchIntents("מה החדשות האחרונות על OpenAI?")).toContain("news");
  });

  it("routes world overview to topics mode", () => {
    expect(isTopicsOverviewQuery("מה קורה בעולם?")).toBe(true);
    expect(isTopicsOverviewQuery("חפש חדשות על חלל")).toBe(false);
  });

  it("normalizes Hebrew topic before engine search", async () => {
    searchNewsMock.mockResolvedValue([]);
    buildRecentHeadlineHitsMock.mockResolvedValue([
      {
        article: {
          id: "1",
          title: "NASA launch",
          source: "NASA",
          url: "https://example.com/1",
        },
        cluster: null,
        score: 20,
        sourceKind: "headline",
      },
    ]);

    const result = await fetchGroveeNewsSearch("חפש חדשות על חלל");

    expect(normalizeNewsEngineQuery("חפש חדשות על חלל")).toBe("space");
    expect(searchNewsMock).toHaveBeenCalledWith("space");
    expect(result.ok).toBe(true);
    expect(result.provider).toBe("grovee-news");
    expect(getNewsPanelPayload()?.cards.length).toBeGreaterThan(0);
  });

  it("topics overview sets panel payload with cards", async () => {
    fetchTopicsBundleMock.mockResolvedValue({
      generatedAt: Date.now(),
      cards: [
        {
          id: "t1",
          title: "World headline",
          titleOriginal: "World headline",
          source: "BBC",
          sourceKey: "bbc",
          url: "https://example.com/t1",
          image: "",
          score: 10,
          publishedTs: Date.now(),
          laneId: "world",
          laneLabel: "World",
          laneIcon: "🌍",
          query: "world",
          matchLabel: "high",
        },
      ],
      stats: { totalLanes: 40, lanesWithHits: 12 },
    });

    const result = await fetchGroveeNewsSearch("מה קורה בעולם?");

    expect(result.ok).toBe(true);
    expect(getNewsPanelPayload()?.mode).toBe("topics");
    expect(getNewsPanelPayload()?.cards).toHaveLength(1);
  });

  it("falls back to recent headlines when search returns empty", async () => {
    searchNewsMock.mockResolvedValue([]);
    buildRecentHeadlineHitsMock.mockResolvedValue([
      {
        article: {
          id: "2",
          title: "Fallback headline",
          source: "Reuters",
          url: "https://example.com/2",
        },
        cluster: null,
        score: 8,
        sourceKind: "headline",
      },
    ]);

    const result = await fetchGroveeNewsSearch("חדשות על איראן");

    expect(buildRecentHeadlineHitsMock).toHaveBeenCalled();
    expect(result.ok).toBe(true);
    expect(getNewsPanelPayload()?.cards[0]?.title).toBe("Fallback headline");
  });
});
