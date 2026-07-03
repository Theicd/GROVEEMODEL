import { describe, expect, it } from "vitest";
import { formatNeoEta } from "./neoEta";
import { parseHorizonsObserverRows } from "./parseHorizonsResult";
import { auToLd, parseCadDistAu } from "./jplApi";
import { buildNeoTrackFromHorizons, raDecDistToLatLon, visualRadiusFromDistAu } from "./neoTrack";
import { getEventSeverity } from "./severityScore";
import type { GlobeAlertEvent } from "./types";

const SAMPLE_SOE = `
$$SOE
 2026-Jul-03 12:00     12 30 00.00 +10 15 30.0    18.0   0.05  0.05200000000000   -5.2000000
 2026-Jul-03 18:00     12 32 15.00 +11 20 45.0    17.5   0.04  0.04100000000000   -4.8000000
$$EOE
`;

describe("parseHorizonsObserverRows", () => {
  it("parses RA/Dec and range from Horizons block", () => {
    const rows = parseHorizonsObserverRows(SAMPLE_SOE);
    expect(rows.length).toBe(2);
    expect(rows[0].distAu).toBeCloseTo(0.052, 4);
    expect(rows[0].deldotKmS).toBeCloseTo(-5.2, 1);
    expect(rows[0].raHours).toBeCloseTo(12.5, 2);
  });
});

describe("neoTrack", () => {
  it("maps RA/Dec to lat/lon", () => {
    const { lat, lon } = raDecDistToLatLon(12, 10);
    expect(lat).toBeCloseTo(10, 0);
    expect(lon).toBeGreaterThanOrEqual(-180);
    expect(lon).toBeLessThan(180);
  });

  it("scales visual radius by lunar distance", () => {
    expect(visualRadiusFromDistAu(auToLd(1) * (384_400 / 149_597_870.7))).toBeGreaterThan(1.2);
    expect(visualRadiusFromDistAu(0.0279)).toBeGreaterThan(2);
  });

  it("builds track with closest point", () => {
    const track = buildNeoTrackFromHorizons("2024 AB1", [
      { timeLabel: "a", raHours: 10, decDeg: 5, distAu: 0.08, deldotKmS: -3, t: 1 },
      { timeLabel: "b", raHours: 10.5, decDeg: 6, distAu: 0.04, deldotKmS: -2, t: 2 },
      { timeLabel: "c", raHours: 11, decDeg: 7, distAu: 0.09, deldotKmS: 2, t: 3 },
    ]);
    expect(track?.closest.distAu).toBe(0.04);
    expect(track?.points.length).toBe(3);
  });
});

describe("formatNeoEta", () => {
  it("formats future approach in hours or days", () => {
    const in6h = Date.now() + 6 * 3_600_000;
    expect(formatNeoEta(in6h)).toMatch(/שע/);
    const in5d = Date.now() + 5 * 86_400_000;
    expect(formatNeoEta(in5d)).toMatch(/ימים/);
  });
});

describe("severityScore neo", () => {
  it("scores close approaches higher", () => {
    const close: GlobeAlertEvent = {
      id: "1",
      type: "neo",
      lat: 0,
      lon: 0,
      location: "test",
      time: Date.now(),
      source: "nasa-jpl",
      distLd: 0.4,
    };
    const far: GlobeAlertEvent = { ...close, distLd: 25 };
    expect(getEventSeverity(close).tier).toBe("critical");
    expect(getEventSeverity(far).score).toBeLessThan(getEventSeverity(close).score);
  });
});
