import { describe, expect, it } from "vitest";
import { mixRssBySource } from "./mixRssBySource";
import type { RssItem } from "../types";

function item(sourceKey: string, n: number): RssItem {
  return {
    id: `${sourceKey}::${n}`,
    title: `${sourceKey} ${n}`,
    description: "",
    source: sourceKey,
    sourceKey,
    category: "world",
    link: `https://example.com/${sourceKey}/${n}`,
    image: "",
    published: new Date().toISOString(),
    publishedTs: n,
    guid: `${n}`,
  };
}

describe("mixRssBySource", () => {
  it("alternates sources round-robin", () => {
    const mixed = mixRssBySource([
      item("a", 1),
      item("a", 2),
      item("b", 1),
      item("c", 1),
      item("b", 2),
    ]);

    expect(mixed.map((i) => i.sourceKey)).toEqual(["a", "b", "c", "a", "b"]);
  });
});

describe("latestRssPerSource", () => {
  it("keeps newest headline per source sorted by time", async () => {
    const { latestRssPerSource } = await import("./mixRssBySource");
    const latest = latestRssPerSource([
      item("a", 1),
      item("a", 2),
      item("b", 1),
      item("c", 1),
    ]);
    expect(latest.map((i) => i.sourceKey).sort()).toEqual(["a", "b", "c"]);
    expect(latest.find((i) => i.sourceKey === "a")?.id).toBe("a::2");
  });
});
