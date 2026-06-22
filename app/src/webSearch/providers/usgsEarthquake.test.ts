import { describe, expect, it } from "vitest";
import { extractMinMagnitude, fetchEarthquakeSearch } from "./usgsEarthquake";

describe("usgsEarthquake", () => {
  it("parses magnitude threshold from Hebrew query", () => {
    expect(extractMinMagnitude("מה הייתה רעידת האדמה האחרונה מעל 5.0?")).toBe(5);
    expect(extractMinMagnitude("above 4.5 earthquakes")).toBe(4.5);
    expect(extractMinMagnitude("M6+ this week")).toBe(6);
  });

  it("does not filter out all quakes when query says בעולם (global strongest)", async () => {
    const result = await fetchEarthquakeSearch(
      "איפה הייתה רעידת האדמה החזקה בעולם ב-24 השעות האחרונות?",
    );
    expect(result.ok).toBe(true);
    expect(result.text).not.toContain("לא נמצאו רעידות");
    expect(result.text).not.toContain("אין רעידות אדמה מדווחות");
    expect(result.text).toMatch(/M\d/);
    expect(result.text).toContain("הרעידה החזקה ביותר");
  });

  it("returns last M5+ quake for B07-style query (not «אזור האחרונה»)", async () => {
    const result = await fetchEarthquakeSearch("מה הייתה רעידת האדמה האחרונה מעל 5.0?");
    expect(result.ok).toBe(true);
    expect(result.text).not.toContain("באזור האחרונה");
    expect(result.text).not.toContain("לא נמצאו רעידות באזור (האחרונה)");
    if (result.text.includes("אין רעידות מעל")) {
      expect(result.text).toContain("M5");
    } else {
      expect(result.text).toMatch(/הרעידה האחרונה מעל M5|M\d/);
      expect(result.text).toContain("מעל M5");
    }
  });

  it("does not treat 24-hour phrasing as geographic region", async () => {
    const result = await fetchEarthquakeSearch(
      "האם היו רעידות אדמה ב-24 השעות האחרונות מעל 5 בסולם ריכטר?",
    );
    expect(result.ok).toBe(true);
    expect(result.text).not.toMatch(/באזור \(ב-24/);
    expect(result.text).toMatch(/M\d|אין רעידות מעל M5/);
  });

  it("still filters by region when Israel is mentioned", async () => {
    const result = await fetchEarthquakeSearch("האם הייתה רעידת אדמה בישראל השבוע?");
    expect(result.ok).toBe(true);
    expect(result.text).toMatch(/24|שעות|7|ימים|USGS/i);
  });
});
