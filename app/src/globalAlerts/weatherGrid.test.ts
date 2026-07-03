import { describe, expect, it } from "vitest";
import { buildWeatherSampleGrid, classifyWeatherCell } from "./weatherGrid";

describe("weatherGrid", () => {
  it("builds coarse global grid", () => {
    const grid = buildWeatherSampleGrid(20);
    expect(grid.length).toBeGreaterThan(100);
  });

  it("classifies thunder codes", () => {
    expect(classifyWeatherCell(95, 0, 40)).toBe("thunder");
    expect(classifyWeatherCell(99, 0, 40)).toBe("thunder");
  });

  it("classifies rain from precipitation", () => {
    expect(classifyWeatherCell(1, 0.4, 20)).toBe("rain");
  });

  it("classifies cloudy from cloud cover", () => {
    expect(classifyWeatherCell(2, 0, 70)).toBe("cloudy");
  });
});
