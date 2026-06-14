import { describe, expect, it } from "vitest";
import { buildDataAgeLines, formatDataAgeForSource } from "./dataAge";
import type { SearchSourceResult } from "./types";

describe("dataAge", () => {
  it("flags stale Frankfurter date", () => {
    const source: SearchSourceResult = {
      provider: "frankfurter-fx",
      label: "FX",
      ok: true,
      text: "תאריך: 2026-06-12\n1 USD = 2.92 ILS",
      latencyMs: 1,
    };
    const line = formatDataAgeForSource(source);
    expect(line).toMatch(/DATA AGE/);
    expect(line).toMatch(/2026-06-12/);
    expect(buildDataAgeLines([source]).length).toBeGreaterThan(0);
  });

  it("flags stale Yahoo market date", () => {
    const source: SearchSourceResult = {
      provider: "yahoo-finance",
      label: "Yahoo",
      ok: true,
      text: "S&P 500: 7431\nעדכון (Yahoo Finance): 2026-06-12 20:54:28 UTC",
      latencyMs: 1,
    };
    const line = formatDataAgeForSource(source);
    expect(line).toMatch(/DATA AGE/);
    expect(line).toMatch(/סגירת מסחר/);
  });
});
