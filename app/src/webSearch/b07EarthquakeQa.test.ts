import { describe, expect, it } from "vitest";
import { buildCapabilityLiveReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { classifySearchIntents, isEarthquakeQuery, isNewsQuery } from "./intents";
import type { SearchSourceResult } from "./types";

const B07 = "מה הייתה רעידת האדמה האחרונה מעל 5.0?";

const usgsSource = (): SearchSourceResult => ({
  provider: "usgs-earthquake",
  label: "רעידות אדמה (USGS)",
  ok: true,
  text: [
    'סה"כ 3 רעידות מעל M5 ב-7 ימים (USGS).',
    "הרעידה האחרונה מעל M5: M5.2 · 10 km S of Kastrí, Greece · 2026-06-20 09:37:24 UTC",
    "מסונן: magnitude ≥ M5.",
    "- M5.2 · 10 km S of Kastrí, Greece · 2026-06-20 09:37:24 UTC",
  ].join("\n"),
  latencyMs: 40,
});

describe("B07 earthquake QA path", () => {
  it("routes to earthquake + news, not pure news query", () => {
    const intents = classifySearchIntents(B07);
    expect(intents).toContain("earthquake");
    expect(isEarthquakeQuery(B07)).toBe(true);
    expect(isNewsQuery(B07)).toBe(false);
  });

  it("builds canned reply with M5.2 from USGS", () => {
    const intents = classifySearchIntents(B07);
    const sources = [usgsSource(), {
      provider: "grovee-news",
      label: "GROVEE NEWS",
      ok: true,
      text: "ANSWER (headline): [ScienceDaily] headline",
      latencyMs: 10,
    }];
    const reply = buildCapabilityLiveReply(B07, intents, sources);
    expect(reply).toMatch(/M5\.2/);
    expect(reply).toMatch(/Greece|Kastr/i);
    expect(reply).not.toMatch(/כרטיסיות/);
  });

  it("shouldDeliverStructuredLiveReply true even without prebuilt canned", () => {
    const intents = classifySearchIntents(B07);
    expect(shouldDeliverStructuredLiveReply(B07, intents, [usgsSource()], null)).toBe(true);
  });

  it("simulated App shouldDeliverLive when canned missing but USGS ok", () => {
    const intents = classifySearchIntents(B07);
    const sources = [usgsSource()];
    let marineLiveCannedReply: string | null = null;
    const searchLiveOk = sources.some((s) => s.ok && s.text.trim());
    const newsQueryTurn = isNewsQuery(B07);
    if (
      !marineLiveCannedReply &&
      shouldDeliverStructuredLiveReply(B07, intents, sources, null)
    ) {
      marineLiveCannedReply = buildCapabilityLiveReply(B07, intents, sources);
    }
    const shouldDeliverLive =
      !!marineLiveCannedReply &&
      !newsQueryTurn &&
      searchLiveOk &&
      shouldDeliverStructuredLiveReply(B07, intents, sources, marineLiveCannedReply);
    expect(shouldDeliverLive).toBe(true);
    expect(marineLiveCannedReply).toMatch(/M5\.2/);
  });
});
