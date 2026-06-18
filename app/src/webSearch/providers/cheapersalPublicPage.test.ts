import { describe, expect, it } from "vitest";
import { applyPublicMetaToHit, parseCheapersalProductHtml } from "./cheapersalPublicPage";
import type { ProductSerpHit } from "../types";

const sampleHtml = `
<meta property="og:image" content="https://price-api.additlist.com/images/catalog/victory/7290000042015.jpg" />
<span>מחיר ממוצע</span><span>₪7.30</span>
<span>המחיר הזול</span><span>₪5.90</span>
`;

const baseHit = (): ProductSerpHit => ({
  id: "catalog-7290000042015",
  barcode: "7290000042015",
  title: "חלב 3% מהדרין",
  url: "https://cheapersal.co.il/product/7290000042015",
  snippet: "מקרר · ברקוד 7290000042015",
  imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/7290000042015.jpg",
  source: "Cheapersal",
});

describe("cheapersalPublicPage", () => {
  it("detects Cheapersal not-found pages", async () => {
    const meta = parseCheapersalProductHtml(
      '<html><title>מוצר לא נמצא | Cheapersal</title></html>',
    );
    expect(meta.imageUrl).toBeUndefined();
    expect(meta.priceNis).toBeUndefined();
  });

  it("parses JSON-LD Product offers (Cheapersal SSR)", () => {
    const html = `<script type="application/ld+json">{
      "@context":"https://schema.org",
      "@type":"Product",
      "image":"https://price-api.additlist.com/images/catalog/carrefour/7290004131074.jpg",
      "offers":{"@type":"AggregateOffer","lowPrice":5.9,"highPrice":7.35}
    }</script>`;
    const meta = parseCheapersalProductHtml(html);
    expect(meta.imageUrl).toContain("7290004131074");
    expect(meta.priceNis).toBe(5.9);
    expect(meta.priceMaxNis).toBe(7.35);
  });

  it("parses og:image and cheapest price from product HTML", () => {
    const meta = parseCheapersalProductHtml(sampleHtml);
    expect(meta.imageUrl).toContain("victory/7290000042015");
    expect(meta.priceNis).toBe(5.9);
    expect(meta.priceAvgNis).toBe(7.3);
  });

  it("merges public meta into product hit", () => {
    const merged = applyPublicMetaToHit(baseHit(), parseCheapersalProductHtml(sampleHtml));
    expect(merged.priceNis).toBe(5.9);
    expect(merged.imageUrl).toContain("victory/");
    expect(merged.priceSummary).toContain("₪5.90");
    expect(merged.snippet).toContain("₪5.90");
  });
});
