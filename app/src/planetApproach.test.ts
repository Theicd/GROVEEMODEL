import { describe, expect, it } from "vitest";
import { planetEase, planetEaseDwell } from "./planetApproach";

describe("planetEaseDwell", () => {
  it("opens closer than classic ease at t=0", () => {
    expect(planetEaseDwell(0)).toBeGreaterThan(planetEase(0));
    expect(planetEaseDwell(0)).toBeGreaterThanOrEqual(0.5);
  });

  it("holds near hero framing through the middle of the segment", () => {
    expect(planetEaseDwell(0.5)).toBeGreaterThan(0.68);
    expect(planetEaseDwell(0.5)).toBeLessThan(0.76);
    expect(planetEaseDwell(0.7)).toBeGreaterThan(0.68);
    expect(planetEaseDwell(0.7)).toBeLessThan(0.76);
  });

  it("exits toward full ease at end of segment", () => {
    expect(planetEaseDwell(1)).toBeGreaterThan(planetEaseDwell(0.85));
  });
});
