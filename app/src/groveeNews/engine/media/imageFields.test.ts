import { describe, expect, it } from "vitest";
import {
  detectStockProvider,
  hasImageUrl,
  hasRealImageUrl,
  isStockImageUrl,
  normalizeImageUrl,
} from "./imageFields";

describe("imageFields", () => {
  it("coerces non-string image values safely", () => {
    expect(normalizeImageUrl(" https://x.test/a.jpg ")).toBe("https://x.test/a.jpg");
    expect(normalizeImageUrl(123 as unknown as string)).toBe("123");
    expect(normalizeImageUrl(null)).toBe("");
    expect(hasImageUrl({})).toBe(false);
    expect(hasImageUrl("https://x.test/a.jpg")).toBe(true);
  });

  it("treats stock hosts as non-real images", () => {
    const pixabay = "https://cdn.pixabay.com/photo/2024/test.jpg";
    expect(detectStockProvider(pixabay)).toBe("pixabay");
    expect(isStockImageUrl(pixabay)).toBe(true);
    expect(hasRealImageUrl(pixabay)).toBe(false);
    expect(hasRealImageUrl("https://cdn.example.com/news/hero.jpg")).toBe(true);
  });
});