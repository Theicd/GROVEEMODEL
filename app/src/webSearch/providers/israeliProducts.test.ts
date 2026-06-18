import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchIsraeliProductsSearch } from "./israeliProducts";

vi.mock("../fetchJson", () => ({
  fetchJson: vi.fn(),
}));

vi.mock("./cheapersalPrices", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./cheapersalPrices")>();
  return {
    ...actual,
    enrichProductHitsWithPrices: vi.fn(async (hits: import("../types").ProductSerpHit[]) => hits),
    isCheapersalConfigured: vi.fn(() => false),
  };
});

import { fetchJson } from "../fetchJson";
import { enrichProductHitsWithPrices, isCheapersalConfigured } from "./cheapersalPrices";

const mockFetch = vi.mocked(fetchJson);
const mockEnrichPrices = vi.mocked(enrichProductHitsWithPrices);
const mockCheapersalConfigured = vi.mocked(isCheapersalConfigured);

describe("fetchIsraeliProductsSearch", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockEnrichPrices.mockReset();
    mockEnrichPrices.mockImplementation(async (hits) => hits);
    mockCheapersalConfigured.mockReturnValue(false);
  });

  it("finds milk from local catalog", async () => {
    mockFetch.mockResolvedValue({ products: [] });

    const result = await fetchIsraeliProductsSearch("חלב תנובה");

    expect(result.ok).toBe(true);
    expect(result.provider).toBe("israeli-products");
    expect(result.productHits?.length).toBeGreaterThan(0);
    const top = result.productHits![0];
    expect(top.title).toMatch(/חלב/i);
    expect(top.barcode).toMatch(/^729/);
    expect(top.url).toMatch(/cheapersal\.co\.il\/product\//);
    expect(top.url).not.toMatch(/openfoodfacts/i);
    expect(result.text).toContain(top.barcode);
  });

  it("extracts product term from price question", async () => {
    mockFetch.mockResolvedValue({ products: [] });

    const result = await fetchIsraeliProductsSearch("כמה עולה חלב");

    expect(result.ok).toBe(true);
    expect(result.productHits?.some((h) => /חלב/i.test(h.title))).toBe(true);
    expect(result.text).toMatch(/שאילתה:|חלב/);
  });

  it("looks up barcode directly", async () => {
    const result = await fetchIsraeliProductsSearch("7290004131074");

    expect(result.ok).toBe(true);
    expect(result.productHits?.[0]?.barcode).toBe("7290004131074");
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("merges Open Food Facts results", async () => {
    mockFetch.mockResolvedValue({
      products: [
        {
          code: "7290009999999",
          product_name_he: "לחם אחיד",
          brands: "אחלה",
          image_front_small_url: "https://images.openfoodfacts.org/images/products/small.jpg",
        },
      ],
    });

    const result = await fetchIsraeliProductsSearch("לחם אחיד");

    expect(result.ok).toBe(true);
    expect(result.productHits?.some((h) => h.barcode === "7290009999999")).toBe(true);
    expect(result.productHits?.find((h) => h.barcode === "7290009999999")?.imageUrl).toContain(
      "openfoodfacts",
    );
  });

  it("enriches prices on product browse when Cheapersal configured", async () => {
    mockCheapersalConfigured.mockReturnValue(true);
    mockFetch.mockResolvedValue({ products: [] });

    await fetchIsraeliProductsSearch("חלב");

    expect(mockEnrichPrices).toHaveBeenCalled();
  });

  it("always calls price enrichment for product search", async () => {
    mockCheapersalConfigured.mockReturnValue(false);
    mockFetch.mockResolvedValue({ products: [] });

    await fetchIsraeliProductsSearch("חלב");

    expect(mockEnrichPrices).toHaveBeenCalled();
  });

  it("enriches prices on price queries", async () => {
    mockCheapersalConfigured.mockReturnValue(true);
    mockFetch.mockResolvedValue({ products: [] });
    mockEnrichPrices.mockImplementation(async (hits) =>
      hits.map((h, i) =>
        i === 0
          ? {
              ...h,
              priceNis: 6.9,
              priceMaxNis: 7.5,
              priceAvgNis: 7.1,
              cheapestChain: "רמי לוי",
              priceSummary: "₪6.90 · עד ₪7.50 · ממוצע ₪7.10 · הכי זול: רמי לוי",
              snippet: `₪6.90 · עד ₪7.50 · ${h.snippet}`,
            }
          : h,
      ),
    );

    const result = await fetchIsraeliProductsSearch("כמה עולה חלב");

    expect(mockEnrichPrices).toHaveBeenCalled();
    expect(result.ok).toBe(true);
    expect(result.productHits?.[0]?.priceNis).toBe(6.9);
    expect(result.text).toContain("ANSWER:");
    expect(result.text).toContain("₪6.90");
  });

  it("returns ok with warning when prices unavailable", async () => {
    mockCheapersalConfigured.mockReturnValue(false);
    mockFetch.mockResolvedValue({ products: [] });

    const result = await fetchIsraeliProductsSearch("כמה עולה חלב");

    expect(result.ok).toBe(true);
    expect(result.productHits?.length).toBeGreaterThan(0);
    expect(result.text).toContain("CHEAPERSAL_API_KEY");
  });

  it("fails clearly when no products match", async () => {
    mockFetch.mockResolvedValue({ products: [] });

    const result = await fetchIsraeliProductsSearch("xyznonproduct123");

    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/לא נמצאו מוצרים/);
  });
});
