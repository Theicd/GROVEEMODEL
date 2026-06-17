import { describe, expect, it } from "vitest";
import type { ArticleRecord } from "../types";
import { pickUniqueSourceArticles } from "./uniqueSourcePick";

function article(id: string, sourceKey: string, cat: string, ts: number): ArticleRecord {
  return {
    id,
    url: `https://example.com/${id}`,
    source: sourceKey,
    sourceKey,
    title: `Story ${id}`,
    image: "",
    publishDate: new Date(ts).toISOString(),
    publishedTs: ts,
    articleText: "Body",
    summary: "Summary with enough text for display checks in other modules.",
    keyFacts: [],
    keywords: [],
    entities: [],
    clusterId: id,
    confidence: "LOW",
    fetchedAt: ts,
    summarizedAt: ts,
    feedCategory: cat,
  };
}

describe("pickUniqueSourceArticles", () => {
  it("returns at most one story per RSS sourceKey", () => {
    const now = Date.now();
    const pool = [
      article("a1", "bbc", "world", now),
      article("a2", "bbc", "world", now - 1000),
      article("b1", "reuters", "world", now - 500),
      article("c1", "techcrunch", "technology", now - 2000),
    ];
    const picked = pickUniqueSourceArticles(pool, 20);
    const keys = picked.map((a) => a.sourceKey);
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys).toContain("bbc");
    expect(keys).toContain("reuters");
    expect(keys).not.toContain("a2");
    expect(picked.find((a) => a.id === "a1")?.id).toBe("a1");
  });

  it("caps at limit with all unique sources", () => {
    const now = Date.now();
    const pool = Array.from({ length: 30 }, (_, i) =>
      article(`x${i}`, `source-${i}`, "world", now - i * 1000),
    );
    const picked = pickUniqueSourceArticles(pool, 20);
    expect(picked.length).toBe(20);
    expect(new Set(picked.map((a) => a.sourceKey)).size).toBe(20);
  });
});
