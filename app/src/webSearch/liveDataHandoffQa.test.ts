import { describe, expect, it } from "vitest";
import {
  applySnapshotFallbacks,
  clearLiveWorldSnapshotCache,
  ingestGlobeLivePayload,
} from "../liveWorld";
import { normalizeNewsEngineQuery } from "../groveeNews/newsQueryNormalize";
import { LIVE_DATA_SCENARIOS } from "./liveDataQueryScenarios";
import { resolveLiveDataHandoff } from "./liveDataHandoff";
import { extractMinMagnitude, fetchEarthquakeSearch } from "./providers/usgsEarthquake";
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
import type { SearchSourceResult } from "./types";

function providersMatch(actual: string[], expected: string[]): boolean {
  return expected.every((p) => actual.includes(p));
}

function rssTermsMatch(engineQuery: string, expected: string[]): boolean {
  const blob = engineQuery.toLowerCase();
  return expected.every((t) => blob.includes(t.toLowerCase()));
}

describe("Live data handoff — query → providers → עולם חי", () => {
  for (const spec of LIVE_DATA_SCENARIOS) {
    it(`${spec.id} ${spec.userQuery.slice(0, 52)}`, () => {
      const handoff = resolveLiveDataHandoff(spec.userQuery);

      for (const intent of spec.expectIntents) {
        expect(handoff.intents).toContain(intent);
      }
      expect(providersMatch(handoff.providers, spec.expectProviders)).toBe(true);

      for (const layer of spec.expectLiveWorldLayers) {
        expect(handoff.liveWorldLayers).toContain(layer);
      }

      if (spec.expectRssTerms.length) {
        expect(rssTermsMatch(handoff.rssEngineQuery, spec.expectRssTerms)).toBe(true);
      }

      if (spec.expectMinMagnitude !== undefined) {
        expect(handoff.minMagnitude).toBe(spec.expectMinMagnitude);
      }
    });
  }

  it("earthquake queries enrich with news intent", () => {
    const h = resolveLiveDataHandoff("רעידות אדמה אחרונות");
    expect(h.intents).toContain("earthquake");
    expect(h.intents).toContain("news");
  });

  it("M5+ query adds disaster intent", () => {
    const q = "האם היו רעידות אדמה ב-24 השעות האחרונות מעל 5 בסולם ריכטר?";
    const h = resolveLiveDataHandoff(q);
    expect(h.intents).toContain("disaster");
    expect(extractMinMagnitude(q)).toBe(5);
  });
});

describe("עולם חי — snapshot history for search fallback", () => {
  it("globe payload feeds earthquake fallback with magnitude and place", () => {
    clearLiveWorldSnapshotCache();
    ingestGlobeLivePayload({
      earthquake: {
        items: [
          { magnitude: 6.1, place: "near coast of Japan", time: Date.now() - 3_600_000 },
          { magnitude: 4.2, place: "Dead Sea region", time: Date.now() - 7_200_000 },
        ],
      },
    });

    const out = applySnapshotFallbacks(
      "רעידות מעל 5?",
      ["earthquake"],
      [{ provider: "usgs-earthquake", label: "x", ok: false, text: "", error: "net", latencyMs: 1 }],
    );
    const eq = out.find((s) => s.provider === "usgs-earthquake" && s.ok);
    expect(eq?.text).toContain("M6.1");
    expect(eq?.text).toContain("Japan");
    expect(eq?.label).toMatch(/עולם חי|USGS/i);
  });

  it("normalizeNewsEngineQuery maps earthquake chat to RSS terms", () => {
    expect(normalizeNewsEngineQuery("האם היו רעידות אדמה הלילה?")).toBe("earthquake");
  });
});

describe("SEARCH BRIEF — sensor + RSS for Gemma", () => {
  const usgsSource: SearchSourceResult = {
    provider: "usgs-earthquake",
    label: "רעידות אדמה (USGS)",
    ok: true,
    text: [
      'סה"כ 3 רעידות מעל M5 ב-24 שעות (USGS).',
      "הרעידה החזקה ביותר: M6.2 · near coast of Japan",
      "- M6.2 · near coast of Japan · 2026-06-15 08:00 UTC",
      "- M5.1 · Chile · 2026-06-15 02:00 UTC",
    ].join("\n"),
    url: "https://earthquake.usgs.gov",
    latencyMs: 50,
  };

  const newsSource: SearchSourceResult = {
    provider: "grovee-news",
    label: "חדשות (GROVEE NEWS)",
    ok: true,
    text: [
      "ANSWER (headline): [BBC] Major earthquake strikes Japan coast",
      "מקור: GROVEE NEWS (סריקת RSS)",
      "[BBC] 1. Major earthquake strikes Japan coast",
      "[Reuters] 2. Tsunami warning after strong quake",
    ].join("\n"),
    latencyMs: 200,
  };

  it("builds brief with USGS facts and news headlines", () => {
    const intents = ["earthquake", "news", "disaster"] as const;
    const brief = buildSearchBrief([usgsSource, newsSource], [...intents], LIVE_DATA_SCENARIOS[1].userQuery);
    expect(brief.intents).toContain("earthquake");
    expect(brief.facts.some((f) => /M6\.2|6\.2/.test(f))).toBe(true);
    expect(brief.facts.some((f) => /Japan|BBC|earthquake/i.test(f))).toBe(true);
  });

  it("formatSearchBriefContext includes ANSWER earthquake + SENSOR+RSS instruction", () => {
    const intents = ["earthquake", "news"] as const;
    const brief = buildSearchBrief([usgsSource, newsSource], [...intents], LIVE_DATA_SCENARIOS[1].userQuery);
    const ctx = formatSearchBriefContext(brief, LIVE_DATA_SCENARIOS[1].userQuery, 2000, [
      usgsSource,
      newsSource,
    ]);
    expect(ctx).toContain("ANSWER (earthquake)");
    expect(ctx).toContain("SENSOR+RSS");
    expect(ctx).toContain("ANSWER (news");
    expect(ctx).toContain("LIVE WORLD");
    expect(ctx).toMatch(/M6\.2|6\.2/);
  });
});

describe("USGS live fetch — magnitude filter (network)", () => {
  it("LD-EQ02 user query returns structured USGS text", async () => {
    const q = LIVE_DATA_SCENARIOS.find((s) => s.id === "LD-EQ02")!.userQuery;
    const result = await fetchEarthquakeSearch(q);
    expect(result.ok).toBe(true);
    expect(result.text).toMatch(/M\d|אין רעידות מעל M5|סה"כ/);
    if (result.text.includes('סה"כ')) {
      const mags = [...result.text.matchAll(/M([\d.]+)/g)].map((m) => parseFloat(m[1]));
      expect(mags.every((m) => m >= 5)).toBe(true);
    }
  });
});
