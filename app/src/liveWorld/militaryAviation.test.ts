import { describe, expect, it } from "vitest";
import { classifyAircraft, isAwacsSuspect } from "../realityData/aviationClassifier";
import {
  buildMilitaryAviationText,
  formatMilitaryAviationCannedReply,
  isAwacsQuery,
  isMilitaryAviationQuery,
} from "./militaryAviation";
import { setLiveWorldSnapshot, clearLiveWorldSnapshotCache } from "./snapshotStore";

describe("aviationClassifier", () => {
  it("marks NATO hex as military", () => {
    const cls = classifyAircraft("e4abc1", "NATO01", "NATO", null);
    expect(cls.mil).toBe(true);
    expect(cls.awacsSuspect).toBe(true);
  });

  it("detects AWACS callsign", () => {
    expect(isAwacsSuspect("SENTRY01", "", null)).toBe(true);
  });
});

describe("military aviation query detection", () => {
  it("detects Hebrew AWACS spellings", () => {
    expect(isAwacsQuery("כמה מטוסי אוואקס פעילים?")).toBe(true);
    expect(isAwacsQuery("כמה מטוסי AWACS פעילים?")).toBe(true);
  });

  it("detects Hebrew military aircraft queries", () => {
    expect(isMilitaryAviationQuery("כמה מטוסים צבאיים מעל ישראל?")).toBe(true);
    expect(isMilitaryAviationQuery("כמה מטוסים מעל ישראל?")).toBe(false);
  });
});

describe("formatMilitaryAviationCannedReply", () => {
  it("formats AWACS zero count in Hebrew", () => {
    const reply = formatMilitaryAviationCannedReply(
      "כמה מטוסי AWACS פעילים כרגע?",
      "ANSWER (AWACS): 0 מטוסים מזוהים כ-AWACS במעקב עולם חי (זיהוי heuristic — לא כל AWACS משדר ADS-B).",
    );
    expect(reply).toMatch(/^0 מטוסי AWACS/);
    expect(reply).not.toMatch(/ANSWER \(AWACS\)/);
  });
});

describe("militaryAviation from snapshot", () => {
  it("counts AWACS suspects from live world cache", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "globe",
      aviation: {
        count: 3,
        regionLabel: "עולם חי",
        sample: [],
        items: [
          { callsign: "ELY123", isMilitary: false },
          { callsign: "NATO01", icao24: "e4abc1", isMilitary: true, milLabel: "NATO", awacsSuspect: true },
          { callsign: "BAW60", isMilitary: false },
        ],
      },
    });

    const text = buildMilitaryAviationText("כמה מטוסי AWACS פעילים כרגע?", {
      fetchedAt: Date.now(),
      source: "globe",
      aviation: {
        count: 3,
        regionLabel: "עולם חי",
        sample: [],
        items: [
          { callsign: "ELY123", isMilitary: false },
          { callsign: "NATO01", icao24: "e4abc1", isMilitary: true, milLabel: "NATO", awacsSuspect: true },
          { callsign: "BAW60", isMilitary: false },
        ],
      },
    });

    expect(text).toContain("ANSWER (AWACS)");
    expect(text).toContain("1");
    expect(text).toContain("NATO01");
  });
});
