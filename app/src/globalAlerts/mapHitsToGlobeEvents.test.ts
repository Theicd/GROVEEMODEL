import { describe, expect, it } from "vitest";
import { hitToGlobeEvent, hitsToGlobeEvents } from "./mapHitsToGlobeEvents";
import type { UnifiedSearchHit } from "../searchResults/types";

describe("mapHitsToGlobeEvents", () => {
  it("maps earthquake hit with coordinates", () => {
    const hit: UnifiedSearchHit = {
      id: "eq-1",
      kind: "earthquake",
      title: "M5.3 · Greece",
      url: "https://earthquake.usgs.gov/x",
      snippet: "2026-07-02 UTC",
      sourceLabel: "USGS",
      provider: "usgs-earthquake",
      publishedTs: Date.now(),
      meta: { magnitude: 5.3, lat: 35.5, lon: 24.2, depth: 10 },
      summarizable: false,
    };
    const ev = hitToGlobeEvent(hit);
    expect(ev?.type).toBe("earthquake");
    expect(ev?.lat).toBe(35.5);
    expect(ev?.lon).toBe(24.2);
  });

  it("skips hits without coordinates", () => {
    const hit: UnifiedSearchHit = {
      id: "eq-2",
      kind: "earthquake",
      title: "M4 · Unknown",
      url: "https://x",
      snippet: "",
      sourceLabel: "USGS",
      provider: "usgs-earthquake",
      meta: { magnitude: 4 },
      summarizable: false,
    };
    expect(hitToGlobeEvent(hit)).toBeNull();
  });

  it("maps weak earthquakes when min mag is zero", () => {
    const hit: UnifiedSearchHit = {
      id: "eq-3",
      kind: "earthquake",
      title: "M2.1 · Test",
      url: "https://x",
      snippet: "",
      sourceLabel: "USGS",
      provider: "usgs-earthquake",
      publishedTs: Date.now(),
      meta: { magnitude: 2.1, lat: 1, lon: 2 },
      summarizable: false,
    };
    expect(hitToGlobeEvent(hit)?.magnitude).toBe(2.1);
  });

  it("maps GDACS hurricane and skips GDACS EQ", () => {
    const hurricane: UnifiedSearchHit = {
      id: "gd-1",
      kind: "disaster",
      title: "Cyclone ALBERTO",
      url: "https://gdacs.org",
      snippet: "Cuba",
      sourceLabel: "GDACS",
      provider: "gdacs-disasters",
      meta: { disasterType: "TC", alertLevel: "Orange", lat: 22, lon: -79 },
      summarizable: false,
    };
    const eqDup: UnifiedSearchHit = {
      id: "gd-2",
      kind: "disaster",
      title: "Earthquake Turkey",
      url: "https://gdacs.org",
      snippet: "Turkey",
      sourceLabel: "GDACS",
      provider: "gdacs-disasters",
      meta: { disasterType: "EQ", lat: 38, lon: 27 },
      summarizable: false,
    };
    const out = hitsToGlobeEvents([hurricane, eqDup]);
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe("hurricane");
  });
});
