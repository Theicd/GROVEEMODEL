import { describe, expect, it } from "vitest";
import { deepVisionBackoffMs, detectVisionBudget } from "./visionBudget";

describe("visionBudget", () => {
  it("detectVisionBudget returns a profile with tier", () => {
    const p = detectVisionBudget();
    expect(["low", "normal"]).toContain(p.tier);
    expect(p.pollIntervalMs).toBeGreaterThan(0);
  });

  it("deepVisionBackoffMs scales with failures", () => {
    const p = detectVisionBudget();
    expect(deepVisionBackoffMs(p, 0)).toBe(0);
    expect(deepVisionBackoffMs(p, 2)).toBeGreaterThan(deepVisionBackoffMs(p, 1));
  });
});
