import { describe, expect, it } from "vitest";
import {
  cheapersalCatalogImage,
  cheapersalProductUrl,
  productImageCandidates,
  shufersalSearchUrl,
  supermarketProductUrl,
} from "./israeliProductLinks";

describe("israeliProductLinks", () => {
  it("uses Cheapersal price-compare page by default", () => {
    expect(cheapersalProductUrl("7290004131074")).toBe(
      "https://cheapersal.co.il/product/7290004131074",
    );
    expect(supermarketProductUrl("7290004131074")).toBe(
      "https://cheapersal.co.il/product/7290004131074",
    );
  });

  it("never links to Open Food Facts", () => {
    const url = supermarketProductUrl("7290010723065");
    expect(url).not.toMatch(/openfoodfacts/i);
  });

  it("routes to chain store when cheapest chain is known", () => {
    expect(supermarketProductUrl("7290004131074", "רמי לוי")).toMatch(/rami-levy\.co\.il/i);
    expect(supermarketProductUrl("7290004131074", "שופרסל")).toBe(shufersalSearchUrl("7290004131074"));
  });

  it("prefers additlist CDN for product images", () => {
    expect(cheapersalCatalogImage("7290004131074")).toContain("additlist.com");
    const candidates = productImageCandidates("7290004131074");
    expect(candidates[0]).toMatch(/additlist\.com/i);
    expect(candidates.length).toBeGreaterThan(2);
  });
});
