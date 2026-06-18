/**
 * Live QA — scrape Cheapersal product page URL for image + price.
 * Run: npm run test:products
 */
import { describe, expect, it } from "vitest";

import { fetchCheapersalPageMeta, parseCheapersalProductHtml } from "./providers/cheapersalPublicPage";
import { enrichProductHitsWithPrices } from "./providers/cheapersalPrices";

describe("Cheapersal page scrape live", () => {
  it("parses milk product page JSON-LD for price and image", async () => {
    const meta = await fetchCheapersalPageMeta("https://cheapersal.co.il/product/7290004131074");
    expect(meta?.imageUrl, JSON.stringify(meta)).toMatch(/additlist|openfoodfacts/i);
    expect(meta?.priceNis).toBeGreaterThan(0);
    expect(meta?.priceMaxNis).toBeGreaterThanOrEqual(meta!.priceNis!);
  }, 20_000);

  it("enriches catalog milk hit from live product page", async () => {
    const [hit] = await enrichProductHitsWithPrices(
      [
        {
          id: "catalog-7290004131074",
          barcode: "7290004131074",
          title: "חלב 3% — תנובה",
          url: "https://cheapersal.co.il/product/7290004131074",
          snippet: "מקרר",
          source: "Cheapersal",
        },
      ],
      1,
    );
    expect(hit.priceNis).toBeGreaterThan(0);
    expect(hit.imageUrl).toBeTruthy();
    expect(hit.priceSummary).toContain("₪");
  }, 25_000);

  it("returns empty meta for missing Cheapersal product", async () => {
    const meta = parseCheapersalProductHtml('<title>מוצר לא נמצא | Cheapersal</title>');
    expect(meta.priceNis).toBeUndefined();
    expect(meta.imageUrl).toBeUndefined();
  });
});
