import { describe, expect, it } from "vitest";

import {
  isGenericCatalogImage,
  openFoodFactsImageCandidates,
  sortProductHitsByRichness,
  stripGenericCatalogImage,
} from "./productImageResolve";

describe("productImageResolve", () => {
  it("detects generic additlist catalog URLs", () => {
    expect(isGenericCatalogImage("https://price-api.additlist.com/images/catalog/carrefour/729.jpg")).toBe(
      true,
    );
    expect(isGenericCatalogImage("https://images.openfoodfacts.org/images/products/x.jpg")).toBe(false);
  });

  it("strips generic catalog guesses", () => {
    expect(stripGenericCatalogImage("https://price-api.additlist.com/images/catalog/carrefour/1.jpg")).toBe(
      undefined,
    );
    expect(stripGenericCatalogImage("https://images.openfoodfacts.org/x.jpg")).toBe(
      "https://images.openfoodfacts.org/x.jpg",
    );
  });

  it("builds OFF image paths from barcode", () => {
    const urls = openFoodFactsImageCandidates("7290004131074");
    expect(urls[0]).toContain("729/000/413/1074");
  });

  it("sorts products with price and image first", () => {
    const sorted = sortProductHitsByRichness([
      { barcode: "a", priceNis: undefined, imageUrl: undefined },
      { barcode: "b", priceNis: 5.9, imageUrl: "https://images.openfoodfacts.org/a.jpg" },
      { barcode: "c", priceNis: 4, imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/x.jpg" },
    ]);
    expect(sorted[0].barcode).toBe("b");
  });
});
