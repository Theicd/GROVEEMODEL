import { describe, expect, it } from "vitest";
import {
  passesRealtimeEarthquakeFilter,
  pickClosestNeoAlert,
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

describe("pickClosestNeoAlert", () => {
  const mkNeo = (id: string, distLd: number, approachOffsetMs: number): GlobeAlertEvent => ({
    id,
    type: "neo",
    lat: 0,
    lon: 0,
    location: id,
    time: Date.now(),
    source: "nasa-jpl",
    distLd,
    approachTime: Date.now() + approachOffsetMs,
  });

  it("picks smallest LD", () => {
    const a = mkNeo("a", 13.2, 3_600_000);
    const b = mkNeo("b", 17.3, 1_800_000);
    expect(pickClosestNeoAlert([a, b])?.id).toBe("a");
  });

  it("breaks LD ties by soonest approach", () => {
    const a = mkNeo("a", 10, 5_000_000);
    const b = mkNeo("b", 10, 2_000_000);
    expect(pickClosestNeoAlert([a, b])?.id).toBe("b");
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
