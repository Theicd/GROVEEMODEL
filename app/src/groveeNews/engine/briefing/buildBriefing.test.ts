import { describe, expect, it } from "vitest";
import { buildDailyBriefing } from "./buildBriefing";
import { pickUniqueSourceArticles } from "../feed/uniqueSourcePick";
import type { ArticleRecord } from "../types";

function article(id: string, cat: string, sourceKey: string, ts: number): ArticleRecord {
  return {
    id,
    url: `https://example.com/${id}`,
    source: sourceKey,
    sourceKey,
    title: `Story ${id}`,
    image: "",
    publishDate: new Date(ts).toISOString(),
    publishedTs: ts,
    articleText: "Body text for testing summaries with enough length to pass quality checks.",
    summary: "A clear summary with enough detail for the briefing builder to include this item.",
    keyFacts: ["Fact one", "Fact two"],
    keywords: [cat],
    entities: [],
    clusterId: id,
    confidence: "LOW",
    fetchedAt: ts,
    summarizedAt: ts,
    feedCategory: cat,
    intelSource: "rss",
  };
}

describe("buildDailyBriefing", () => {
  it("is exported as async function", () => {
    expect(typeof buildDailyBriefing).toBe("function");
  });
});

describe("briefing unique RSS sources", () => {
  it("pickUniqueSourceArticles never repeats sourceKey", () => {
    const now = Date.now();
    const items = [
      article("w1", "world", "bbc", now),
      article("w2", "world", "bbc", now - 1000),
      article("t1", "technology", "wired", now - 2000),
      article("a1", "ai", "openai_blog", now - 3000),
    ];
    const picked = pickUniqueSourceArticles(items, 20);
    const keys = picked.map((a) => a.sourceKey);
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys.filter((k) => k === "bbc")).toHaveLength(1);
  });
});
