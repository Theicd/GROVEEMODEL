import { describe, expect, it } from "vitest";
import {
  clearSharedRegionCache,
  extractRegionPhrase,
  shouldResolveSharedRegion,
} from "./sharedRegion";

describe("sharedRegion", () => {
  it("detects cross-source queries needing shared geocode", () => {
    expect(shouldResolveSharedRegion("האם יש סופה באזור ישראל וגם מטוסים", ["weather", "aviation"])).toBe(true);
    expect(shouldResolveSharedRegion("מה מזג האוויר בתל אביב", ["weather"])).toBe(false);
  });

  it("extracts region phrase from aliases", () => {
    expect(extractRegionPhrase("מזג אוויר בישראל ומטוסים מעל")).toBe("ישראל");
    expect(extractRegionPhrase("ships near Haifa bay")).toBe("Haifa");
  });

  it("clears cache without error", () => {
    clearSharedRegionCache();
    expect(true).toBe(true);
  });
});
