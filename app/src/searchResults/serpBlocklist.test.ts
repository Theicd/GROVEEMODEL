import { describe, expect, it } from "vitest";
import {
  filterBlockedHits,
  isBlockedSerpHit,
  isBlockedSerpUrl,
} from "./serpBlocklist";
import type { UnifiedSearchHit } from "./types";

describe("serpBlocklist", () => {
  it("blocks The Verge Calvin and Hobbes deal URL", () => {
    expect(
      isBlockedSerpUrl(
        "https://www.theverge.com/gadgets/950958/the-complete-calvin-and-hobbes-fathers-day-gift",
      ),
    ).toBe(true);
  });

  it("blocks verge gadget deal slugs", () => {
    expect(isBlockedSerpUrl("https://www.theverge.com/gadgets/950929/paramount-plus-two-month-deal-sale")).toBe(
      true,
    );
  });

  it("blocks mis-parsed promo titles", () => {
    const hit: UnifiedSearchHit = {
      id: "x",
      kind: "rss",
      title: "פרסומות",
      url: "https://www.theverge.com/gadgets/123/foo",
      snippet: "",
      sourceLabel: "The Verge",
      provider: "grovee-news",
      summarizable: false,
    };
    expect(isBlockedSerpHit(hit)).toBe(true);
  });

  it("keeps normal verge tech articles", () => {
    expect(isBlockedSerpUrl("https://www.theverge.com/tech/123456/some-real-story")).toBe(false);
  });

  it("blocks GhanaWeb RSS from SERP", () => {
    expect(isBlockedSerpUrl("https://www.ghanaweb.com/GhanaHomePage/sports/archive/1")).toBe(true);
    expect(
      isBlockedSerpHit({
        title: "Ghana headline",
        url: "https://www.ghanaweb.com/foo",
        sourceLabel: "GhanaWeb",
        sourceKey: "gh_ghanaweb",
      }),
    ).toBe(true);
  });

  it("filterBlockedHits removes blocked entries", () => {
    const hits: UnifiedSearchHit[] = [
      {
        id: "bad",
        kind: "rss",
        title: "Calvin and Hobbes Father's Day gift",
        url: "https://www.theverge.com/gadgets/950958/gift",
        snippet: "",
        sourceLabel: "The Verge",
        provider: "grovee-news",
        summarizable: false,
      },
      {
        id: "ok",
        kind: "rss",
        title: "Real headline",
        url: "https://www.bbc.com/news/world-123",
        snippet: "",
        sourceLabel: "BBC",
        provider: "grovee-news",
        summarizable: true,
      },
    ];
    expect(filterBlockedHits(hits)).toHaveLength(1);
    expect(filterBlockedHits(hits)[0].id).toBe("ok");
  });
});
