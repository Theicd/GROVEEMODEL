import { describe, expect, it } from "vitest";
import {
  passesRealtimeEarthquakeFilter,
  sortAlertEvents,
} from "./alertFilters";
import { EQ_LIVE_WINDOW_MS } from "./types";
import type { GlobeAlertEvent } from "./types";

const base: GlobeAlertEvent = {
  id: "1",
  type: "earthquake",
  lat: 0,
  lon: 0,
  location: "test",
  time: Date.now() - 5 * 60_000,
  source: "usgs",
  magnitude: 3.2,
};

describe("passesRealtimeEarthquakeFilter", () => {
  it("accepts recent quakes of any magnitude", () => {
    expect(passesRealtimeEarthquakeFilter(base)).toBe(true);
    expect(passesRealtimeEarthquakeFilter({ ...base, magnitude: 1.2 })).toBe(true);
  });

  it("rejects earthquakes older than the live window", () => {
    expect(
      passesRealtimeEarthquakeFilter({
        ...base,
        time: Date.now() - EQ_LIVE_WINDOW_MS - 60_000,
      }),
    ).toBe(false);
  });
});

describe("sortAlertEvents", () => {
  it("puts earth events above neo", () => {
    const eq: GlobeAlertEvent = { ...base, id: "eq" };
    const neo: GlobeAlertEvent = {
      id: "neo",
      type: "neo",
      lat: 0,
      lon: 0,
      location: "neo",
      time: Date.now(),
      source: "nasa-jpl",
      distLd: 5,
      approachTime: Date.now() + 3600000,
    };
    const sorted = [neo, eq].sort(sortAlertEvents);
    expect(sorted[0].type).toBe("earthquake");
    expect(sorted[1].type).toBe("neo");
  });
});
