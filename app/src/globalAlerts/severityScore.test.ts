import { describe, expect, it } from "vitest";
import { getEventSeverity } from "./severityScore";
import type { GlobeAlertEvent } from "./types";

const base = (over: Partial<GlobeAlertEvent>): GlobeAlertEvent => ({
  id: "x",
  type: "earthquake",
  lat: 0,
  lon: 0,
  location: "test",
  time: Date.now(),
  source: "usgs",
  ...over,
});

describe("getEventSeverity", () => {
  it("rates M4.7 as low", () => {
    expect(getEventSeverity(base({ magnitude: 4.7 })).tier).toBe("low");
  });

  it("rates M5.2 as moderate", () => {
    expect(getEventSeverity(base({ magnitude: 5.2 })).tier).toBe("moderate");
  });

  it("rates M6.2 as high", () => {
    expect(getEventSeverity(base({ magnitude: 6.2 })).tier).toBe("high");
  });

  it("rates M7+ as critical", () => {
    expect(getEventSeverity(base({ magnitude: 7.1 })).tier).toBe("critical");
  });

  it("rates hurricane cat 4 as critical", () => {
    expect(
      getEventSeverity(base({ type: "hurricane", category: 4, source: "gdacs" })).tier,
    ).toBe("critical");
  });

  it("rates hurricane cat 3 as high", () => {
    expect(
      getEventSeverity(base({ type: "hurricane", category: 3, source: "gdacs" })).tier,
    ).toBe("high");
  });
});
