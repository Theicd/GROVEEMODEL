import { describe, expect, it } from "vitest";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
import { classifySearchIntents } from "./intents";
import type { SearchSourceResult } from "./types";

const B11 = "כמה אוניות נמצאות כרגע בתעלת סואץ?";

const suezShipsSource = (): SearchSourceResult => ({
  provider: "ais-ships",
  label: "ספינות (AIS / עולם חי)",
  ok: true,
  text: [
    "אזור: תעלת סואץ",
    "ANSWER (ships live): 0",
    "דיווח AIS חי + עולם חי: 0 (0 AIS · 0 עולם חי)",
    "סימוני מסלול (הדגמה — לא AIS חי): 2",
    "הערה: בתעלת סואץ ובאזורים מחוץ לצפון אירופה אין כיסוי AIS חי מ-Digitraffic — הספירה 0.",
    "1. Suez Transit · מסלול (הדגמה) · 31.25,32.31 · —",
  ].join("\n"),
  url: "https://meri.digitraffic.fi",
  latencyMs: 20,
});

describe("B11 Suez ships QA path", () => {
  it("builds count-first canned reply", () => {
    const intents = classifySearchIntents(B11);
    const reply = buildCapabilityLiveReply(B11, intents, [suezShipsSource()]);
    expect(reply).toMatch(/^0 אוניות בתעלת סואץ לפי AIS/);
    expect(reply).not.toMatch(/הדגמה|מספר ספינות|several/i);
  });

  it("shouldDeliverStructuredLiveReply true for ok ais-ships", () => {
    const intents = classifySearchIntents(B11);
    expect(shouldDeliverStructuredLiveReply(B11, intents, [suezShipsSource()], null)).toBe(true);
  });

  it("search brief includes ANSWER ships count and GAPS", () => {
    const intents = classifySearchIntents(B11);
    const sources = [suezShipsSource()];
    const brief = buildSearchBrief(sources, intents, B11, undefined, "count");
    const ctx = formatSearchBriefContext(brief, B11, 900, sources, "count");
    expect(ctx).toMatch(/ANSWER \(ships\)/);
    expect(ctx).toMatch(/GAPS: אין דיווח AIS/);
  });
});
