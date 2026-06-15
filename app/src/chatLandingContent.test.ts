import { describe, expect, it } from "vitest";
import {
  LANDING_CAPABILITY_CHIPS,
  LANDING_ROTATION_MS,
  labelWordCount,
  pickRotatingLandingSuggestions,
} from "./chatLandingContent";

describe("chatLandingContent", () => {
  it("has organized capability pool", () => {
    expect(LANDING_CAPABILITY_CHIPS.length).toBeGreaterThan(30);
    const cats = new Set(LANDING_CAPABILITY_CHIPS.map((c) => c.category));
    expect(cats.size).toBeGreaterThan(8);
  });

  it("every label has at least 3 words", () => {
    for (const chip of LANDING_CAPABILITY_CHIPS) {
      expect(labelWordCount(chip.label)).toBeGreaterThanOrEqual(3);
    }
  });

  it("pickRotatingLandingSuggestions returns 3 from different categories when possible", () => {
    const picks = pickRotatingLandingSuggestions(3);
    expect(picks).toHaveLength(3);
    for (const pick of picks) {
      expect(labelWordCount(pick.label)).toBeGreaterThanOrEqual(3);
    }
    const uniqueCats = new Set(picks.map((p) => p.category));
    expect(uniqueCats.size).toBeGreaterThanOrEqual(2);
  });

  it("rotation interval is 10 seconds", () => {
    expect(LANDING_ROTATION_MS).toBe(10_000);
  });
});
