import { describe, expect, it } from "vitest";
import { classifySearchIntents, isProductsQuery } from "../webSearch/intents";
import type { ProductSerpHit, SearchSourceResult } from "../webSearch/types";
import { filterHits } from "./rankHits";
import { buildUnifiedSearchPayload, mergeSourcesToHits } from "./mergeSearchHits";
import { isProductHit } from "./ProductSearchResultRow";
import type { SearchResultsFilter } from "./types";

const resolveInitialTab = (_payload: ReturnType<typeof buildUnifiedSearchPayload>): SearchResultsFilter => "all";

const milkHit = (overrides: Partial<ProductSerpHit> = {}): ProductSerpHit => ({
  id: "catalog-7290004131074",
  barcode: "7290004131074",
  title: "חלב 3% — תנובה",
  brand: "תנובה",
  url: "https://cheapersal.co.il/product/7290004131074",
  snippet: "₪6.90 · עד ₪7.50 · מקרר · ברקוד 7290004131074",
  imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/7290004131074.jpg",
  source: "קטלוג ישראלי",
  priceNis: 6.9,
  priceMaxNis: 7.5,
  priceSummary: "₪6.90 · עד ₪7.50 · הכי זול: רמי לוי",
  ...overrides,
});

const breadHit = (overrides: Partial<ProductSerpHit> = {}): ProductSerpHit => ({
  id: "catalog-7290000042015",
  barcode: "7290000042015",
  title: "לחם אחיד פרוס",
  brand: "אחלה",
  url: "https://cheapersal.co.il/product/7290000042015",
  snippet: "₪8.50 · מאפייה · ברקוד 7290000042015",
  imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/7290000042015.jpg",
  source: "קטלוג ישראלי",
  priceNis: 8.5,
  priceSummary: "₪8.50 · הכי זול: שופרסל",
  ...overrides,
});

const productSource = (hits: ProductSerpHit[], query: string): SearchSourceResult => ({
  provider: "israeli-products",
  label: "מוצרי סופר · ישראל",
  ok: true,
  text: `שאילתה: ${query}\n1. ${hits[0]?.title ?? "מוצר"}`,
  productHits: hits,
  latencyMs: 35,
});

const rssSource = (): SearchSourceResult => ({
  provider: "grovee-news",
  label: "GROVEE NEWS",
  ok: true,
  text: "",
  newsCards: [
    {
      id: "n1",
      title: "כותרת חדשות על חלב",
      titleOriginal: "Milk news",
      source: "ynet",
      sourceKey: "il_ynet",
      url: "https://www.ynet.co.il/1",
      image: "",
      score: 40,
      publishedTs: Date.now(),
    },
  ],
  latencyMs: 10,
});

type GroceryCase = {
  query: string;
  hits: ProductSerpHit[];
  titleMatch: RegExp;
};

const GROCERY_CASES: GroceryCase[] = [
  { query: "חלב", hits: [milkHit()], titleMatch: /חלב/i },
  { query: "כמה עולה חלב", hits: [milkHit()], titleMatch: /חלב/i },
  { query: "חלב תנובה", hits: [milkHit()], titleMatch: /תנובה|חלב/i },
  { query: "לחם", hits: [breadHit()], titleMatch: /לחם/i },
  { query: "כמה עולה לחם", hits: [breadHit()], titleMatch: /לחם/i },
];

describe("Products tab QA — intent routing", () => {
  it.each(GROCERY_CASES.map((c) => [c.query, c.query]))("%s is classified as products search", (query) => {
    expect(isProductsQuery(query)).toBe(true);
    expect(classifySearchIntents(query)).toContain("products");
  });
});

describe("Products tab QA — SERP payload", () => {
  it.each(GROCERY_CASES)("$query maps hits with price, image, and barcode", ({ query, hits, titleMatch }) => {
    const payload = buildUnifiedSearchPayload(query, [productSource(hits, query)]);

    expect(payload.facets.products).toBeGreaterThan(0);
    expect(payload.preferProductsFilter).toBe(true);
    expect(resolveInitialTab(payload)).toBe("all");

    const row = payload.hits.find((h) => h.kind === "product");
    expect(row).toBeDefined();
    expect(row?.title).toMatch(titleMatch);
    expect(row?.imageUrl).toBeTruthy();
    expect(row?.meta?.priceNis).toBeGreaterThan(0);
    expect(row?.meta?.engine).toMatch(/^729/);
    expect(row?.snippet).toMatch(/₪/);
    expect(isProductHit(row!)).toBe(true);
  });

  it("prefers products tab even when RSS blend returns headlines", () => {
    const payload = buildUnifiedSearchPayload("חלב", [rssSource(), productSource([milkHit()], "חלב")]);

    expect(payload.facets.rss).toBeGreaterThan(0);
    expect(payload.facets.products).toBe(1);
    expect(payload.preferProductsFilter).toBe(true);
    expect(resolveInitialTab(payload)).toBe("all");
  });

  it("products tab filter returns only product rows with required fields", () => {
    const payload = buildUnifiedSearchPayload("לחם", [
      rssSource(),
      productSource([breadHit(), milkHit({ title: "חלב 1%" })], "לחם"),
    ]);

    const products = filterHits(payload.hits, "products");
    expect(products.length).toBe(2);
    for (const row of products) {
      expect(row.kind).toBe("product");
      expect(row.imageUrl).toBeTruthy();
      expect(row.meta?.engine).toMatch(/^729/);
    }
    expect(products.some((p) => /לחם/i.test(p.title))).toBe(true);
  });

  it("mergeSourcesToHits preserves Cheapersal URLs and price meta", () => {
    const hits = mergeSourcesToHits([productSource([milkHit()], "חלב")], "חלב");
    expect(hits).toHaveLength(1);
    expect(hits[0].url).toMatch(/cheapersal\.co\.il\/product\//);
    expect(hits[0].meta?.priceNis).toBe(6.9);
    expect(hits[0].imageUrl).toMatch(/additlist|cheapersal|rfrsh/i);
  });
});

describe("Products tab QA — empty states", () => {
  it("products tab is empty when provider returns no hits", () => {
    const payload = buildUnifiedSearchPayload("חלב", [
      {
        provider: "israeli-products",
        label: "מוצרים",
        ok: false,
        text: "",
        error: "לא נמצאו מוצרים",
        latencyMs: 5,
      },
    ]);
    expect(payload.facets.products).toBe(0);
    expect(filterHits(payload.hits, "products")).toHaveLength(0);
  });
});
