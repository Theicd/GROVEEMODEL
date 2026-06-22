import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RssItem } from "./engine/types";

const getAllRssItemsMock = vi.fn();

vi.mock("./engine/storage/db", () => ({
  getAllRssItems: (...args: unknown[]) => getAllRssItemsMock(...args),
}));

vi.mock("./engine/settings/userNewsProfile", () => ({
  getUserNewsProfile: () => ({ locale: "he-IL", uiLanguage: "he", pollTier: "core" }),
}));

const sampleItems = (): RssItem[] => [
  {
    id: "il-1",
    link: "https://www.ynet.co.il/1",
    source: "ynet",
    sourceKey: "il_ynet",
    title: "ממשלה דנה בחקיקה",
    description: "פוליטיקה",
    guid: "il-1",
    published: new Date().toISOString(),
    publishedTs: Date.now(),
    image: "",
    category: "news",
  },
  {
    id: "en-1",
    link: "https://www.bbc.com/1",
    source: "BBC",
    sourceKey: "bbc",
    title: "World economy update",
    description: "Markets",
    guid: "en-1",
    published: new Date().toISOString(),
    publishedTs: Date.now() - 1000,
    image: "",
    category: "news",
  },
];

describe("buildRecentHeadlineHits", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAllRssItemsMock.mockResolvedValue(sampleItems());
  });

  it("returns ranked headlines for specific Hebrew news topics", async () => {
    const { buildRecentHeadlineHits } = await import("./recentHeadlineHits");
    const hits = await buildRecentHeadlineHits("חדשות פוליטיקה ישראל");
    expect(hits.length).toBeGreaterThan(0);
  });

  it("returns recent headlines for non-news panel queries when RSS DB has items", async () => {
    const { buildRecentHeadlineHits } = await import("./recentHeadlineHits");
    const hits = await buildRecentHeadlineHits("מכונית חשמלית");
    expect(hits.length).toBeGreaterThan(0);
  });

  it("returns empty when RSS DB is empty", async () => {
    getAllRssItemsMock.mockResolvedValue([]);
    const { buildRecentHeadlineHits } = await import("./recentHeadlineHits");
    const hits = await buildRecentHeadlineHits("חדשות");
    expect(hits).toHaveLength(0);
  });
});
