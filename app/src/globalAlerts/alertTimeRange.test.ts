import { describe, expect, it } from "vitest";
import { filterEventsForRange } from "./alertTimeRange";
import { passesSpacePanelNeo } from "./alertFilters";
import type { GlobeAlertEvent } from "./types";

const eq = (mag: number, ageMs: number): GlobeAlertEvent => ({
  id: `eq-${mag}-${ageMs}`,
  type: "earthquake",
  lat: 0,
  lon: 0,
  location: "test",
  time: Date.now() - ageMs,
  source: "usgs",
  magnitude: mag,
});

const storm = (alert: string): GlobeAlertEvent => ({
  id: `storm-${alert}`,
  type: "hurricane",
  lat: 0,
  lon: 0,
  location: "BAVI-26 · Guam",
  time: Date.now(),
  source: "gdacs",
  alertLevel: alert,
  gdacsIsCurrent: true,
  updatedTime: Date.now(),
});

describe("filterEventsForRange", () => {
  it("live tab keeps recent quakes and active GDACS", () => {
    const now = Date.now();
    const events = [
      eq(6.2, 10 * 60_000),
      eq(6.1, 40 * 60_000),
      storm("Red"),
    ];
    const out = filterEventsForRange(events, "live", now);
    expect(out).toHaveLength(2);
    expect(out.some((e) => e.type === "hurricane")).toBe(true);
  });

  it("space tab returns only neo events", () => {
    const now = Date.now();
    const events: GlobeAlertEvent[] = [
      eq(6, 0),
      {
        id: "neo-1",
        type: "neo",
        lat: 0,
        lon: 0,
        location: "NEO",
        time: now,
        source: "nasa-jpl",
        approachTime: now + 3_600_000,
        distLd: 10,
      },
    ];
    const out = filterEventsForRange(events, "space", now);
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe("neo");
  });

  it("passesSpacePanelNeo rejects showcase catalog entries", () => {
    const now = Date.now();
    const showcase: GlobeAlertEvent = {
      id: "neo-showcase-eros",
      type: "neo",
      lat: 0,
      lon: 0,
      location: "433 Eros",
      time: now,
      source: "nasa-jpl",
      approachTime: now + 60_000,
      distLd: 12,
      showcaseNeo: true,
    };
    expect(passesSpacePanelNeo(showcase)).toBe(false);
  });

  it("passesSpacePanelNeo rejects past flybys", () => {
    const now = Date.now();
    const past: GlobeAlertEvent = {
      id: "neo-past",
      type: "neo",
      lat: 0,
      lon: 0,
      location: "past",
      time: now,
      source: "nasa-jpl",
      approachTime: now - 60_000,
      distLd: 5,
    };
    const future: GlobeAlertEvent = {
      ...past,
      id: "neo-future",
      approachTime: now + 60_000,
    };
    expect(passesSpacePanelNeo(past)).toBe(false);
    expect(passesSpacePanelNeo(future)).toBe(true);
  });
});
