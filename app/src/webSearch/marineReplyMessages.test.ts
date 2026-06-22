import { describe, expect, it } from "vitest";
import { buildMarineLiveReply, formatShipsCannedReply } from "./marineReplyMessages";
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
      "עודכן: 2026-06-13 16:30:00 UTC",
    ].join("\n"),
    latencyMs: 50,
  };

  const rotterdamShips: SearchSourceResult = {
    provider: "ais-ships",
    label: "ספינות",
    ok: true,
    text: [
      "אזור: נמל רוטרדם",
      "ANSWER (ships live): 2",
      "דיווח AIS חי + עולם חי: 2 (2 AIS · 0 עולם חי)",
      "עודכן: 2026-06-13 16:30:00 UTC",
      "1. MAERSK · AIS · 51.92,4.48 · 5.0 kn",
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

  it("builds short Haifa zero-count reply without demo markers", () => {
    const reply = buildMarineLiveReply("כמה אוניות במפרץ חיפה?", ["ships"], [haifaShips]);
    expect(reply).toMatch(/^0 אוניות במפרץ חיפה לפי AIS/);
    expect(reply).not.toMatch(/הדגמה|Suez|Haifa Cargo|מסלול/i);
    expect(reply!.split("\n").length).toBeLessThanOrEqual(3);
  });

  it("builds short Suez zero-count reply", () => {
    const reply = buildMarineLiveReply("כמה אוניות נמצאות כרגע בתעלת סואץ?", ["ships"], [suezShips]);
    expect(reply).toMatch(/^0 אוניות בתעלת סואץ לפי AIS/);
    expect(reply).not.toMatch(/הדגמה|Digitraffic|REALITY/i);
  });

  it("includes live ship samples when count > 0", () => {
    const reply = formatShipsCannedReply("כמה אוניות ליד רוטרדם?", rotterdamShips.text);
    expect(reply).toMatch(/^2 אוניות/);
    expect(reply).toMatch(/MAERSK/);
  });

  it("builds buoys canned reply from Overpass", () => {
    const reply = buildMarineLiveReply("כמה מצופים במפרץ חיפה?", ["marine-infra"], [haifaBuoys]);
    expect(reply).toMatch(/3 מצופים/);
    expect(reply).toMatch(/OpenStreetMap/);
  });
});
