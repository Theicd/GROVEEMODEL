import { describe, expect, it } from "vitest";
import { getHurricaneIntensity, parseWindKmh, windToCategory } from "./hurricaneIntensity";

describe("hurricaneIntensity", () => {
  it("parses wind from GDACS severity text", () => {
    expect(parseWindKmh("Hurricane (maximum wind speed of 185 km/h)")).toBe(185);
  });

  it("uses higher category when wind exceeds alert proxy", () => {
    const int = getHurricaneIntensity(2, 220);
    expect(int.category).toBeGreaterThanOrEqual(4);
    expect(int.spinSpeed).toBeGreaterThan(0.44);
  });

  it("cat 5 spins faster than cat 1", () => {
    const c1 = getHurricaneIntensity(1);
    const c5 = getHurricaneIntensity(5);
    expect(c5.spinSpeed).toBeGreaterThan(c1.spinSpeed);
    expect(c5.maxRadius).toBeGreaterThan(c1.maxRadius);
    expect(c5.color).not.toBe(c1.color);
  });

  it("maps wind to category", () => {
    expect(windToCategory(260)).toBe(5);
    expect(windToCategory(130)).toBe(1);
  });
});
