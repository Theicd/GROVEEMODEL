import { describe, expect, it } from "vitest";
import { slicePage } from "./buildFeedPagination";

describe("buildFeed pagination", () => {
  it("slicePage returns a window and hasMore flag", () => {
    const items = Array.from({ length: 50 }, (_, i) => ({
      kind: "article" as const,
      id: `a-${i}`,
      sortTs: 50 - i,
      article: {} as never,
    }));

    const page0 = slicePage(items, 0, 30);
    expect(page0.items).toHaveLength(30);
    expect(page0.hasMore).toBe(true);
    expect(page0.nextOffset).toBe(30);

    const page1 = slicePage(items, 30, 30);
    expect(page1.items).toHaveLength(20);
    expect(page1.hasMore).toBe(false);
    expect(page1.nextOffset).toBe(60);
  });
});
