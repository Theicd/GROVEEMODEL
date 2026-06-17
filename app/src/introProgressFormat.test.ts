import { describe, expect, it } from "vitest";

import {
  downloadProgressPercent,
  formatDownloadPercent,
  normalizeHfProgressPercent,
  resolveDownloadPercent,
  sumFileProgressMap,
} from "./introProgressFormat";

describe("introProgressFormat", () => {
  it("computes byte ratio percent", () => {
    expect(downloadProgressPercent(390_000_000, 3_900_000_000)).toBeCloseTo(10, 5);
  });

  it("formats fractional percent below 1%", () => {
    expect(formatDownloadPercent(0.05)).toBe("0.05");
    expect(formatDownloadPercent(42)).toBe("42");
  });

  it("normalizes HF ratio vs percent", () => {
    expect(normalizeHfProgressPercent(0.715)).toBeCloseTo(71.5, 5);
    expect(normalizeHfProgressPercent(42)).toBe(42);
  });

  it("prefers bytes over per-file HF progress=100", () => {
    const loaded = 2_620_000_000;
    const total = 3_660_000_000;
    expect(resolveDownloadPercent({ loaded, total, hfProgress: 100 })).toBeCloseTo(
      downloadProgressPercent(loaded, total),
      4,
    );
    expect(resolveDownloadPercent({ loaded, total, hfProgress: 100 })).toBeLessThan(100);
  });

  it("sums multi-file progress map", () => {
    const sum = sumFileProgressMap({
      "a.onnx": { loaded: 100, total: 200 },
      "b.onnx": { loaded: 50, total: 100 },
    });
    expect(sum).toEqual({ loaded: 150, total: 300 });
  });
});
