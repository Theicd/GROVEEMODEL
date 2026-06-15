import { describe, expect, it, vi, beforeEach } from "vitest";
import {
  clearStarlinkCatalogCacheForTests,
  fetchStarlinkCatalogSearch,
} from "./satelliteCatalog";

vi.mock("../../webSearch/fetchJson", () => ({
  fetchText: vi.fn(),
}));

import { fetchText } from "../../webSearch/fetchJson";

const mockFetchText = vi.mocked(fetchText);

const tleForCount = (n: number): string => {
  const lines: string[] = [];
  for (let i = 0; i < n; i++) {
    lines.push(`STARLINK-${i}`);
    lines.push(`1 25544U 98067A   08264.51782528 -.000021809  00000-0 -11603-4 0  2927`);
    lines.push(`2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391063537`);
  }
  return lines.join("\n");
};

describe("fetchStarlinkCatalogSearch", () => {
  beforeEach(() => {
    clearStarlinkCatalogCacheForTests();
    mockFetchText.mockReset();
  });

  it("returns ANSWER line with catalog count", async () => {
    mockFetchText.mockResolvedValue(tleForCount(2));

    const result = await fetchStarlinkCatalogSearch("כמה לווייני Starlink פעילים?");
    expect(result.ok).toBe(true);
    expect(result.provider).toBe("starlink-catalog");
    expect(result.text).toContain("ANSWER (Starlink active): 2");
  });

  it("uses cache on second call", async () => {
    mockFetchText.mockResolvedValue(tleForCount(1));
    await fetchStarlinkCatalogSearch("כמה Starlink?");
    await fetchStarlinkCatalogSearch("כמה Starlink?");
    expect(mockFetchText).toHaveBeenCalledTimes(1);
  });

  it("returns seed count when CelesTrak fetch fails", async () => {
    mockFetchText.mockRejectedValue(new Error("timeout"));
    const result = await fetchStarlinkCatalogSearch("כמה לווייני Starlink פעילים?");
    expect(result.ok).toBe(true);
    expect(result.text).toMatch(/ANSWER \(Starlink active\): \d+/);
  });
});

describe("isStarlinkCountQuery", () => {
  it("classifies B13 prompt", async () => {
    const { isStarlinkCountQuery, isStarlinkRegionalQuery } = await import("../../webSearch/intents");
    expect(isStarlinkCountQuery("כמה לווייני Starlink פעילים כרגע?")).toBe(true);
    expect(isStarlinkRegionalQuery("כמה לווייני Starlink פעילים כרגע?")).toBe(false);
  });
});
