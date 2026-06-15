import { describe, expect, it } from "vitest";
import { buildSearchBrief, formatSearchBriefContext, rerankBriefFacts } from "./searchBrief";
import type { SearchSourceResult } from "./types";

describe("searchBrief", () => {
  it("compresses github results", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "github",
        label: "GitHub",
        ok: true,
        text: "- pytorch/pytorch [Python]: ML ★85000\n- huggingface/transformers ★140000",
        url: "https://github.com",
        latencyMs: 100,
      },
    ];
    const brief = buildSearchBrief(sources, ["github"], "popular github");
    expect(brief.facts.length).toBeGreaterThan(0);
    expect(brief.facts.length).toBeLessThanOrEqual(8);
    const ctx = formatSearchBriefContext(brief, "popular github");
    expect(ctx).toContain("SEARCH BRIEF");
    expect(ctx.length).toBeLessThanOrEqual(900);
  });

  it("includes wind line in weather brief", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "open-meteo",
        label: "מזג אוויר",
        ok: true,
        text: [
          "מיקום: Paris, FR",
          "זמן (מקומי): 2026-06-13T10:00",
          "מצב: מעונן",
          "טמפרatura: 18°C (מרגיש 17°C)",
          "לחות: 60%",
          "רוח: 12.5 km/h, כיוון 180°",
          "לחץ: 1013 hPa",
        ].join("\n"),
        url: "https://open-meteo.com",
        latencyMs: 50,
      },
    ];
    const brief = buildSearchBrief(sources, ["weather"], "מה מהירות הרוח בפריז");
    expect(brief.facts.some((f) => /רוח/i.test(f))).toBe(true);
  });

  it("formats earthquake list for model", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "usgs-earthquake",
        label: "USGS",
        ok: true,
        text: 'סה"כ 42 רעידות ב-24 שעות (USGS). 2 הגדולות:\n- M5.2 · Japan · 2026-06-13 08:00 UTC',
        latencyMs: 80,
      },
    ];
    const brief = buildSearchBrief(sources, ["earthquake"], "רעידות 24 שעות");
    expect(brief.facts[0]).toMatch(/42/);
    expect(brief.facts.some((f) => /M5/i.test(f))).toBe(true);
  });

  it("allows more facts for cross-source intents", () => {
    const sources: SearchSourceResult[] = Array.from({ length: 10 }, (_, i) => ({
      provider: "open-meteo",
      label: `מקור ${i}`,
      ok: true,
      text: `שורה ${i}\nטמפרatura: ${i}°C`,
      latencyMs: 1,
    }));
    const brief = buildSearchBrief(sources, ["weather", "aviation", "disaster"], "הצלבה");
    expect(brief.facts.length).toBeGreaterThan(8);
    expect(brief.facts.length).toBeLessThanOrEqual(14);
  });

  it("adds AWACS ANSWER line when query mentions AWACS", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "adsb-aviation",
        label: "ADS-B",
        ok: true,
        text: "ANSWER (AWACS): 2 · NATO01 · NATO",
        latencyMs: 1,
      },
    ];
    const brief = buildSearchBrief(sources, ["aviation"], "כמה מטוסי AWACS פעילים?");
    const ctx = formatSearchBriefContext(brief, "כמה מטוסי AWACS פעילים?");
    expect(ctx).toContain("ANSWER (AWACS)");
    expect(ctx).toMatch(/NATO|heuristic|AWACS/i);
  });

  it("includes DATA AGE for stale Frankfurter in brief", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "frankfurter-fx",
        label: "Frankfurter",
        ok: true,
        text: "תאריך: 2020-01-01\n1 USD = 3.5 ILS",
        latencyMs: 1,
      },
    ];
    const brief = buildSearchBrief(sources, ["currency"], "מה שער הדולר?");
    const ctx = formatSearchBriefContext(brief, "מה שער הדולר?", 900, sources);
    expect(ctx).toContain("DATA AGE");
  });

  it("prefers PM in government ANSWER when query asks for prime minister", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "wikidata-gov",
        label: "ממשל (Wikidata)",
        ok: true,
        text: [
          "מדינה: United Kingdom (Wikidata Q145)",
          "נושאי משרה (Wikidata):",
          "- צ'ארלס השלישי · ראש מדינה / נשיא",
          "- קיר סטארמר · ראש ממשלה",
          "ANSWER: ראש הממשלה (Wikidata): קיר סטארמר",
        ].join("\n"),
        latencyMs: 1,
      },
    ];
    const brief = buildSearchBrief(sources, ["government"], "מי ראש ממשלת בריטניה?");
    const ctx = formatSearchBriefContext(brief, "מי ראש ממשלת בריטניה?", 900, sources);
    expect(ctx).toContain("ANSWER (government):");
    expect(ctx).toMatch(/ANSWER \(government\):[^\n]*(?:סטארמר|Starmer)/i);
  });

  it("caps github facts per provider", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "github",
        label: "GitHub",
        ok: true,
        text: Array.from({ length: 8 }, (_, i) => `- repo${i}/proj ★${i}`).join("\n"),
        latencyMs: 1,
      },
    ];
    const brief = buildSearchBrief(sources, ["github"], "github repos");
    expect(brief.facts.length).toBeLessThanOrEqual(3);
  });

  it("reranks weather facts ahead for weather intent", () => {
    const facts = [
      "[GitHub] repo/foo",
      "[מזג אוויר] רוח: 20 km/h",
      "[GitHub] repo/bar",
    ];
    const ranked = rerankBriefFacts(facts, ["weather"], "מה הרוח בפריז");
    expect(ranked[0]).toMatch(/רוח/);
  });

  it("includes ANSWER SHAPE and SHARED REGION in context", () => {
    const brief = buildSearchBrief([], ["weather", "aviation"], "cross query", 800, "count");
    const ctx = formatSearchBriefContext(brief, "cross query", 1400, [], "count", "ישראל");
    expect(ctx).toContain("ANSWER SHAPE: count");
    expect(ctx).toContain("SHARED REGION: ישראל");
  });

  it("adds CORRELATION lines for cross-source weather + aviation", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "open-meteo",
        label: "מזג אוויר",
        ok: true,
        text: "מיקום: Israel\nמצב: סופת רעמים\nרוח: 50 km/h",
        latencyMs: 1,
      },
      {
        provider: "adsb-aviation",
        label: "ADS-B",
        ok: true,
        text: "אזור: ישראל\nמטוסים בטווח: 18",
        latencyMs: 1,
      },
    ];
    const brief = buildSearchBrief(sources, ["weather", "aviation"], "האם יש סופה בישראל וגם מטוסים");
    const ctx = formatSearchBriefContext(
      brief,
      "האם יש סופה בישראל וגם מטוסים",
      1400,
      sources,
      "short_fact",
      "ישראל",
    );
    expect(ctx).toContain("CORRELATION");
    expect(ctx).toContain("SHARED REGION: ישראל");
  });
});
