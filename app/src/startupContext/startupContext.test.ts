import { describe, expect, it } from "vitest";
import { isLocalContextTimeQuery, buildLocalTimeAnswer } from "./localTime";
import type { StartupContext } from "./types";

const sampleCtx: StartupContext = {
  fetchedAt: Date.now(),
  datetime: "2026-06-13T10:30:00+03:00",
  timezone: "Asia/Jerusalem",
  utcOffset: "+03:00",
  dst: false,
  dayOfWeek: 5,
  weekNumber: 24,
  countryCode: "IL",
  countryName: "Israel",
  cityName: "Tel Aviv",
  lat: 32.0853,
  lon: 34.7818,
};

describe("isLocalContextTimeQuery", () => {
  it("matches local time without place", () => {
    expect(isLocalContextTimeQuery("מה השעה?")).toBe(true);
    expect(isLocalContextTimeQuery("מה השעה עכשיו")).toBe(true);
    expect(isLocalContextTimeQuery("what time is it")).toBe(true);
    expect(isLocalContextTimeQuery("איזה יום היום")).toBe(true);
  });

  it("does not match foreign place queries", () => {
    expect(isLocalContextTimeQuery("מה השעה בטוקיו")).toBe(false);
    expect(isLocalContextTimeQuery("what time in London")).toBe(false);
    expect(isLocalContextTimeQuery("כמה שעות הפרש בין ישראל ללונדון")).toBe(false);
  });
});

describe("buildLocalTimeAnswer", () => {
  it("includes timezone and place", () => {
    const text = buildLocalTimeAnswer(sampleCtx, "מה השעה?");
    expect(text).toContain("Asia/Jerusalem");
    expect(text).toContain("Tel Aviv");
    expect(text).toContain("[LOCAL CONTEXT");
  });
});
