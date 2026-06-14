import { describe, expect, it } from "vitest";
import { buildMarineLiveReply } from "./marineReplyMessages";
import type { SearchSourceResult } from "./types";

describe("marineReplyMessages", () => {
  const haifaShips: SearchSourceResult = {
    provider: "ais-ships",
    label: "ספינות",
    ok: true,
    text: [
      "אזור: מפרץ חיפה",
      "ANSWER (ships live): 0",
      "דיווח AIS חי + עולם חי: 0 (0 AIS · 0 עולם חי)",
      "סימוני מסלול (הדגמה — לא AIS חי): 2",
      "עודכן: 2026-06-13 16:30:00 UTC",
      "1. Haifa Cargo · מסלול (הדגמה) · 32.82,35.00 · —",
      "2. Haifa Port Route · מסלול (הדגמה) · 32.79,35.02 · —",
    ].join("\n"),
    latencyMs: 50,
  };

  const suezShips: SearchSourceResult = {
    provider: "ais-ships",
    label: "ספינות",
    ok: true,
    text: [
      "אזור: תעלת סואץ",
      "ANSWER (ships live): 0",
      "דיווח AIS חי + עולם חי: 0 (0 AIS · 0 עולם חי)",
      "סימוני מסלול (הדגמה — לא AIS חי): 2",
      "עודכן: 2026-06-13 16:30:00 UTC",
    ].join("\n"),
    latencyMs: 50,
  };

  const haifaBuoys: SearchSourceResult = {
    provider: "osm-overpass-marine",
    label: "OSM",
    ok: true,
    text: [
      "אזור: מפרץ חיפה (OpenStreetMap / Overpass)",
      "תשתיות ימיות בטווח: 3 (0 נמלים · 3 מצופים · 0 מגדלורים · 0 רציפים)",
      "1. buoy A · buoy",
    ].join("\n"),
    latencyMs: 80,
  };

  it("builds Haifa ships canned reply with live count 0", () => {
    const reply = buildMarineLiveReply("כמה כלי שייט במפרץ חיפה?", ["ships"], [haifaShips]);
    expect(reply).toMatch(/ANSWER: 0/);
    expect(reply).toMatch(/מפרץ חיפה/);
    expect(reply).toMatch(/הדגמה/);
  });

  it("builds Suez canned reply — 0 live, not «2 ships»", () => {
    const reply = buildMarineLiveReply("כמה אוניות נמצאות כרגע בתעלת סואץ?", ["ships"], [suezShips]);
    expect(reply).toMatch(/ANSWER: 0/);
    expect(reply).toMatch(/סואץ|תעלת/i);
    expect(reply).not.toMatch(/^ANSWER: 2/m);
  });

  it("builds buoys canned reply from Overpass", () => {
    const reply = buildMarineLiveReply("כמה מצופים במפרץ חיפה?", ["marine-infra"], [haifaBuoys]);
    expect(reply).toMatch(/3 מצופים/);
    expect(reply).toMatch(/OpenStreetMap/);
  });

  it("returns null when provider failed", () => {
    const reply = buildMarineLiveReply("אוניות ליד רוטרדם", ["ships"], [
      { ...haifaShips, ok: false, text: "" },
    ]);
    expect(reply).toBeNull();
  });
});

