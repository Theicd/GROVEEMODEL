import { describe, expect, it } from "vitest";
import {
  parseAisShipsText,
  parseMarineInfraText,
  parseLiveShipCountFromText,
  parseShipSampleLine,
  SERP_SHIP_CARD_CAP,
} from "./liveShipsHits";

const sampleShipsText = `אזור: מפרץ חיפה
ANSWER (ships live): 2
דיווח AIS חי + עולם חי: 2 (1 AIS · 1 עולם חי)
עודכן: 2026-06-15 12:00:00 UTC
1. MSC TEST · AISStream · 32.81,34.96 · 12.3 kn → ROTTERDAM
2. Haifa Cargo · מסלול (הדגמה) · 32.82,34.98 · — kn
3. Globe Ferry · עולם חי · 32.79,34.91 · 5.0 kn`;

const sampleInfraText = `אזור: מפרץ חיפה (OpenStreetMap / Overpass)
תשתיות ימיות בטווח: 3 (1 נמלים · 2 מצופים · 0 מגדלורים · 0 רציפים)
1. Haifa Port · harbour · 32.82,34.98
2. Buoy Alpha · buoy · 32.80,34.95`;

describe("liveShipsHits", () => {
  it("parses live ship lines and excludes demo route markers", () => {
    const hits = parseAisShipsText(sampleShipsText);
    expect(hits.length).toBe(2);
    expect(hits.every((h) => h.kind === "ship")).toBe(true);
    expect(hits.some((h) => /Haifa Cargo/i.test(h.title))).toBe(false);
    expect(hits[0].meta?.shipSource).toBe("aisstream");
    expect(hits[0].meta?.speedKn).toBeCloseTo(12.3);
    expect(hits[1].meta?.shipSource).toBe("globe");
  });

  it("caps ship cards to avoid SERP flooding", () => {
    const lines = ["אזור: test", "ANSWER (ships live): 50"];
    for (let i = 0; i < 70; i++) {
      lines.push(`${i + 1}. Ship ${i} · AIS · 60.1,24.9 · 1.0 kn`);
    }
    const hits = parseAisShipsText(lines.join("\n"));
    expect(hits.length).toBe(SERP_SHIP_CARD_CAP);
  });

  it("parses marine infra samples into marine hits", () => {
    const hits = parseMarineInfraText(sampleInfraText);
    expect(hits.length).toBe(2);
    expect(hits[0].kind).toBe("marine");
    expect(hits[0].meta?.marineInfraKind).toBe("harbour");
    expect(hits[1].meta?.marineInfraKind).toBe("buoy");
  });

  it("reads live count from provider ANSWER line", () => {
    expect(parseLiveShipCountFromText(sampleShipsText)).toBe(2);
  });

  it("parseShipSampleLine rejects malformed lines", () => {
    expect(parseShipSampleLine("not a ship")).toBeNull();
    expect(parseShipSampleLine("1. X · AIS · bad,coords · — kn")).toBeNull();
  });
});

describe("mergeSourcesToHits ships", () => {
  it("builds ship hits from ais-ships provider", async () => {
    const { mergeSourcesToHits, buildUnifiedSearchPayload } = await import("./mergeSearchHits");
    const sources = [
      {
        provider: "ais-ships" as const,
        label: "ספינות",
        ok: true,
        text: sampleShipsText,
        latencyMs: 1,
      },
    ];
    const hits = mergeSourcesToHits(sources, "כמה אוניות במפרץ חיפה");
    expect(hits.filter((h) => h.kind === "ship").length).toBe(2);
    const payload = buildUnifiedSearchPayload("כמה אוניות במפרץ חיפה", sources);
    expect(payload.facets.ships).toBe(2);
    expect(payload.preferShipsFilter).toBe(true);
  });
});
