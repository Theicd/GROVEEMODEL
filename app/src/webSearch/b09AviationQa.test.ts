import { describe, expect, it } from "vitest";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
import { classifySearchIntents } from "./intents";
import type { SearchSourceResult } from "./types";

const B09 = "כמה מטוסים נמצאים כרגע מעל ישראל?";

const liveWorldAviationSource = (): SearchSourceResult => ({
  provider: "adsb-aviation",
  label: "תעופה (עולם חי / ADS-B)",
  ok: true,
  text: [
    "מקור: עולם חי / ADS-B (ישראל (מרכז) · רדיוס 250km)",
    'סה"כ 49 מטוסים במעקב · 10 צבאיים · 0 AWACS? · 1 תדלוק?',
    "כל המטוסים: 49",
    "1. WMT2507 · 34050m",
  ].join("\n"),
  url: "https://api.airplanes.live",
  latencyMs: 20,
});

describe("B09 aviation QA path", () => {
  it("builds canned reply with aircraft count from live world text", () => {
    const intents = classifySearchIntents(B09);
    const reply = buildCapabilityLiveReply(B09, intents, [liveWorldAviationSource()]);
    expect(reply).toMatch(/^49 מטוסים מעל ישראל/);
    expect(reply).toMatch(/10 צבאיים/);
    expect(reply).not.toMatch(/רדיוס 250km/);
    expect(reply).not.toMatch(/אינו זמין/);
  });

  it("shouldDeliverStructuredLiveReply true for ADS-B ok source", () => {
    const intents = classifySearchIntents(B09);
    expect(shouldDeliverStructuredLiveReply(B09, intents, [liveWorldAviationSource()], null)).toBe(true);
  });

  it("search brief includes ANSWER aircraft count for live world facts", () => {
    const intents = classifySearchIntents(B09);
    const sources = [liveWorldAviationSource()];
    const brief = buildSearchBrief(sources, intents, B09, undefined, "count");
    const ctx = formatSearchBriefContext(brief, B09, 900, sources, "count");
    expect(ctx).toMatch(/ANSWER \(aircraft count\)/);
    expect(ctx).toMatch(/49/);
  });
});
