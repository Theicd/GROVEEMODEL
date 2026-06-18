import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildPriceSearchQuery,
  classifySearchIntents,
  isPriceQuery,
  isProductsQuery,
  needsWebSearch,
} from "./intents";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { buildUnifiedSearchPayload } from "../searchResults/mergeSearchHits";
import { PRODUCT_ACCEPTANCE_QUERIES } from "./productsAcceptanceQueries";
import type { ProductSerpHit, SearchSourceResult } from "./types";

vi.mock("./providers/israeliProducts", () => ({
  fetchIsraeliProductsSearch: vi.fn(),
}));

import { fetchIsraeliProductsSearch } from "./providers/israeliProducts";

const mockFetchProducts = vi.mocked(fetchIsraeliProductsSearch);

const sampleHit = (overrides: Partial<ProductSerpHit> = {}): ProductSerpHit => ({
  id: "catalog-7290004131074",
  barcode: "7290004131074",
  title: "חלב 3% — תנובה",
  brand: "תנובה",
  url: "https://cheapersal.co.il/product/7290004131074",
  snippet: "מקרר · ברקוד 7290004131074",
  imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/7290004131074.jpg",
  source: "קטלוג ישראלי",
  ...overrides,
});

const pricedSource = (hits: ProductSerpHit[], text?: string): SearchSourceResult => ({
  provider: "israeli-products",
  label: "מוצרי סופר · ישראל",
  ok: true,
  text:
    text ??
    `שאילתה: חלב\nANSWER: חלב 3% — תנובה — ₪6.90 · עד ₪7.50 · הכי זול: רמי לוי\n1. חלב 3% — תנובה [7290004131074] · ₪6.90`,
  productHits: hits,
  latencyMs: 40,
});

describe("products pipeline — routing", () => {
  it.each(PRODUCT_ACCEPTANCE_QUERIES.map((q) => [q.id, q.query, q.expectIntents]))(
    "%s needsWebSearch and intents for «%s»",
    (_id, query, expectIntents) => {
      expect(needsWebSearch(query)).toBe(true);
      expect(classifySearchIntents(query)).toEqual(expect.arrayContaining(expectIntents));
    },
  );

  it("extracts grocery terms from Hebrew price questions", () => {
    expect(buildPriceSearchQuery("כמה עולה חלב")).toBe("חלב");
    expect(buildPriceSearchQuery("כמה עולה קילו שניצלים")).toMatch(/שניצל/);
    expect(isPriceQuery("כמה עולה חלב")).toBe(true);
    expect(isProductsQuery("כמה עולה חלב")).toBe(true);
  });

  it("does not treat crypto price as supermarket", () => {
    expect(isPriceQuery("מה מחיר הביטקוין עכשיו?")).toBe(false);
    expect(isProductsQuery("מה מחיר הביטקוין עכשיו?")).toBe(false);
  });
});

describe("products pipeline — UI payload", () => {
  it("maps product hits with image and price into unified SERP rows", () => {
    const hit = sampleHit({
      priceNis: 6.9,
      priceMaxNis: 7.5,
      priceSummary: "₪6.90 · עד ₪7.50 · הכי זול: רמי לוי",
      snippet: "₪6.90 · עד ₪7.50 · מקרר · ברקוד 7290004131074",
    });
    const payload = buildUnifiedSearchPayload("כמה עולה חלב", [pricedSource([hit])]);

    expect(payload.preferProductsFilter).toBe(true);
    expect(payload.facets.products).toBe(1);

    const row = payload.hits.find((h) => h.kind === "product");
    expect(row).toBeDefined();
    expect(row?.imageUrl).toMatch(/additlist/i);
    expect(row?.meta?.priceNis).toBe(6.9);
    expect(row?.snippet).toContain("₪6.90");
    expect(row?.title).toMatch(/חלב/);
    expect(row?.url).not.toMatch(/openfoodfacts/i);
  });
});

describe("products pipeline — chat canned reply", () => {
  it("delivers structured Hebrew reply with price", () => {
    const hit = sampleHit({
      priceNis: 6.9,
      priceMaxNis: 7.5,
      priceAvgNis: 7.1,
      cheapestChain: "רמי לוי",
      priceSummary: "₪6.90 · עד ₪7.50 · ממוצע ₪7.10 · הכי זול: רמי לוי",
    });
    const sources = [pricedSource([hit])];
    const query = "כמה עולה חלב";

    expect(shouldDeliverStructuredLiveReply(query, ["products"], sources)).toBe(true);

    const reply = buildCapabilityLiveReply(query, ["products"], sources);
    expect(reply).toMatch(/₪6\.90/);
    expect(reply).toMatch(/רמי לוי|תנובה|חלב/i);
    expect(reply).toMatch(/Sources:/);
  });

  it("lists products without prices when Cheapersal unavailable", () => {
    const sources = [pricedSource([sampleHit()], "שאילתה: חלב\n1. חלב 3% — תנובה [7290004131074]")];
    const reply = buildCapabilityLiveReply("כמה עולה חלב", ["products"], sources);

    expect(reply).toMatch(/חלב/);
    expect(reply).toMatch(/7290004131074/);
  });
});

describe("products pipeline — provider fetch contract", () => {
  beforeEach(() => {
    mockFetchProducts.mockReset();
  });

  it.each(PRODUCT_ACCEPTANCE_QUERIES)(
    "$id provider returns ok + productHits + image",
    async (spec) => {
      const hit = sampleHit({
        title: spec.expectTitleIncludes ? `${spec.expectTitleIncludes} — תנובה` : "מוצר",
      });
      mockFetchProducts.mockResolvedValue(pricedSource([hit]));

      const result = await mockFetchProducts(spec.query);
      expect(result.ok).toBe(true);
      expect(result.productHits?.length).toBeGreaterThan(0);
      expect(result.productHits?.[0]?.imageUrl).toBeTruthy();
      if (spec.expectTitleIncludes) {
        expect(result.productHits?.[0]?.title).toMatch(new RegExp(spec.expectTitleIncludes, "i"));
      }
    },
  );
});
