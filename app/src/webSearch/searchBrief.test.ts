import { describe, expect, it } from "vitest";
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
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
});
