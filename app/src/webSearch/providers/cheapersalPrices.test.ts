import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  enrichProductHitsWithPrices,
  formatProductPriceSummary,
} from "./cheapersalPrices";
import type { ProductSerpHit } from "../types";

vi.mock("../fetchJson", () => ({
  fetchJson: vi.fn(),
}));

vi.mock("./productImageResolve", () => ({
  isGenericCatalogImage: (url?: string) => Boolean(url?.includes("additlist.com/images/catalog/")),
  stripGenericCatalogImage: (url?: string) =>
    url?.includes("additlist.com/images/catalog/") ? undefined : url,
  resolveProductImageUrl: vi.fn(async () => undefined),
  sortProductHitsByRichness: <T,>(hits: T[]) => hits,
}));

vi.mock("./cheapersalPublicPage", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./cheapersalPublicPage")>();
  return {
    ...actual,
    fetchCheapersalPageMeta: vi.fn(async () => null),
    fetchCheapersalPublicMeta: vi.fn(async () => null),
  };
});

import { fetchJson } from "../fetchJson";

const mockFetch = vi.mocked(fetchJson);

const baseHit = (): ProductSerpHit => ({
  id: "catalog-7290004131074",
  barcode: "7290004131074",
  title: "חלב 3% — תנובה",
  url: "https://cheapersal.co.il/product/7290004131074",
  snippet: "מקרר · ברקוד 7290004131074",
  source: "קטלוג ישראלי",
});

describe("cheapersalPrices", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("formats Hebrew price summary", () => {
    const line = formatProductPriceSummary({
      ...baseHit(),
      priceNis: 6.9,
      priceMaxNis: 7.5,
      priceAvgNis: 7.1,
      cheapestChain: "רמי לוי",
      priceStoreCount: 42,
    });
    expect(line).toContain("₪6.90");
    expect(line).toContain("₪7.50");
    expect(line).toContain("רמי לוי");
    expect(line).toContain("42 חנויות");
  });

  it("enriches hits from Cheapersal API response", async () => {
    mockFetch.mockResolvedValue({
      success: true,
      data: {
        product: { unitQty: "1 ליטר" },
        summary: {
          cheapest: 6.9,
          mostExpensive: 7.5,
          average: 7.1,
          storeCount: 30,
          cheapestChain: { name: "רמי לוי" },
        },
      },
    });

    const [priced] = await enrichProductHitsWithPrices([baseHit()], 1);

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/products\/7290004131074\/prices$/),
      undefined,
      expect.any(Object),
    );
    expect(priced.priceNis).toBe(6.9);
    expect(priced.cheapestChain).toBe("רמי לוי");
    expect(priced.url).toMatch(/rami-levy\.co\.il/i);
    expect(priced.priceSummary).toContain("₪6.90");
  });

  it("returns hits unchanged when fetch fails", async () => {
    mockFetch.mockRejectedValue(new Error("HTTP 401"));

    const hits = await enrichProductHitsWithPrices([baseHit()], 1);
    expect(hits[0].priceNis).toBeUndefined();
  });
});
