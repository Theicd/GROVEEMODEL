import { describe, expect, it } from "vitest";
import type { UnifiedSearchHit } from "./types";
import { hitsNeedTranslation } from "./useTranslatedSearchHits";
describe("useTranslatedSearchHits policy", () => {
  const productHit: UnifiedSearchHit = {
    id: "p1",
    kind: "product",
    title: "חלב 3% — תנובה",
    titleOriginal: "חלב 3% — תנובה",
    snippet: "₪6.90",
    snippetOriginal: "₪6.90",
    url: "https://cheapersal.co.il/product/7290004131074",
    sourceLabel: "קטלוג",
    provider: "israeli-products",
    summarizable: false,
    meta: { engine: "7290004131074", priceNis: 6.9 },
    imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/7290004131074.jpg",
  };

  const englishWebHit: UnifiedSearchHit = {
    id: "w1",
    kind: "web",
    title: "Milk nutrition facts",
    snippet: "Calcium and protein overview",
    url: "https://example.com/milk",
    sourceLabel: "Example",
    provider: "wikipedia-en",
    summarizable: false,
  };

  it("skips translation pass when only product hits are present", () => {
    expect(hitsNeedTranslation([productHit], "he")).toBe(false);
  });

  it("still translates mixed product + English web hits", () => {
    expect(hitsNeedTranslation([productHit, englishWebHit], "he")).toBe(true);
  });
});
