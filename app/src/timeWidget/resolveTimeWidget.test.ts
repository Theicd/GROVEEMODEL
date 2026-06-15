import { describe, expect, it } from "vitest";
import {
  buildTimeWidgetFromStartupContext,
  isSinglePlaceTimeWidgetQuery,
} from "./resolveTimeWidget";
import type { StartupContext } from "../startupContext/types";

const sampleCtx: StartupContext = {
  fetchedAt: Date.now(),
  datetime: "2026-06-15T07:16:00+03:00",
  timezone: "Asia/Jerusalem",
  utcOffset: "+03:00",
  dst: false,
  dayOfWeek: 0,
  countryCode: "IL",
  countryName: "Israel",
  cityName: "Jerusalem",
  lat: 31.77,
  lon: 35.21,
};

describe("timeWidget", () => {
  it("detects single-place time queries", () => {
    expect(isSinglePlaceTimeWidgetQuery("מה השעה?")).toBe(true);
    expect(isSinglePlaceTimeWidgetQuery("מה השעה בטוקיו")).toBe(true);
    expect(isSinglePlaceTimeWidgetQuery("כמה שעות הפרש בין ישראל ללונדון")).toBe(false);
  });

  it("builds widget from startup context", () => {
    const widget = buildTimeWidgetFromStartupContext(sampleCtx);
    expect(widget.placeLabel).toContain("Jerusalem");
    expect(widget.timezone).toBe("Asia/Jerusalem");
    expect(widget.anchorIso).toBeTruthy();
  });
});
