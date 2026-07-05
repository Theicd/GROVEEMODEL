import { describe, expect, it } from "vitest";
import { formatAlertCardDisplay } from "./alertCardDisplay";
import type { GlobeAlertEvent } from "./types";

describe("formatAlertCardDisplay", () => {
  it("hurricane shows storm name, category chip, and region", () => {
    const ev: GlobeAlertEvent = {
      id: "tc-1",
      type: "hurricane",
      lat: 0,
      lon: 0,
      location: "MAYSAK-26",
      regionLabel: "Viet Nam, China, Laos",
      time: Date.now(),
      source: "gdacs",
      category: 3,
      alertLevel: "Orange",
      severityText: "Tropical Storm (maximum wind speed of 93 km/h)",
    };
    const d = formatAlertCardDisplay(ev);
    expect(d.headline).toBe("MAYSAK-26");
    expect(d.chips).toContain("קטגוריה 3");
    expect(d.chips).toContain("סערה טропית");
    expect(d.region).toBe("Viet Nam, China, Laos");
    expect(d.detail).toContain("93");
  });

  it("earthquake keeps place in LTR region row", () => {
    const ev: GlobeAlertEvent = {
      id: "eq-1",
      type: "earthquake",
      lat: 0,
      lon: 0,
      location: "58 km W of Tobelo, Indonesia",
      time: Date.now(),
      source: "usgs",
      magnitude: 6.2,
      depth: 58,
    };
    const d = formatAlertCardDisplay(ev);
    expect(d.headline).toBe("M6.2");
    expect(d.region).toBe("58 km W of Tobelo, Indonesia");
    expect(d.regionLtr).toBe(true);
    expect(d.detail).toBe('עומק 58 ק"מ');
  });
});
