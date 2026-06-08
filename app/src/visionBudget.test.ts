import { describe, expect, it } from "vitest";
import { deepVisionBackoffMs, detectVisionBudget } from "./visionBudget";

describe("visionBudget", () => {
  it("detectVisionBudget always returns normal tier (no hardware gating)", () => {
    const p = detectVisionBudget();
    expect(p.tier).toBe("normal");
    expect(p.pollIntervalMs).toBeGreaterThan(0);
  });

  it("deepVisionBackoffMs scales with failures", () => {
    const p = detectVisionBudget();
    expect(deepVisionBackoffMs(p, 0)).toBe(0);
    expect(deepVisionBackoffMs(p, 2)).toBeGreaterThan(deepVisionBackoffMs(p, 1));
  });
});
