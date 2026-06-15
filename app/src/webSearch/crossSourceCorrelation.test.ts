import { describe, expect, it } from "vitest";
import {
  buildCrossSourceCorrelationLines,
  extractCrossSourceMetrics,
} from "./crossSourceCorrelation";
import type { SearchSourceResult } from "./types";

const weatherSource = (text: string): SearchSourceResult => ({
  provider: "open-meteo",
  label: "מזג אוויר",
  ok: true,
  text,
  latencyMs: 1,
});

const aviationSource = (text: string): SearchSourceResult => ({
  provider: "adsb-aviation",
  label: "ADS-B",
  ok: true,
  text,
  latencyMs: 1,
});

describe("crossSourceCorrelation", () => {
  it("extracts weather and aviation metrics", () => {
    const sources = [
      weatherSource(
        ["מיקום: Israel", "מצב: סופת רעמים", "רוח: 55 km/h, כיוון 180°"].join("\n"),
      ),
      aviationSource(["אזור: ישראל (מרכז)", "מטוסים בטווח: 42"].join("\n")),
    ];
    const metrics = extractCrossSourceMetrics(sources, "ישראל");
    expect(metrics.weather?.stormLike).toBe(true);
    expect(metrics.aviation?.count).toBe(42);
  });

  it("builds yes/no correlation for storm + aircraft", () => {
    const metrics = extractCrossSourceMetrics(
      [
        weatherSource("מיקום: Israel\nמצב: סופת רעמים\nרוח: 50 km/h"),
        aviationSource("אזור: ישראל\nמטוסים בטווח: 12"),
      ],
      "ישראל",
    );
    const lines = buildCrossSourceCorrelationLines(
      "האם יש סופה באזור ישראל וגם מטוסים",
      metrics,
      ["weather", "aviation"],
    );
    expect(lines[0]).toMatch(/^CORRELATION:/);
    expect(lines[0]).toMatch(/12|42|כן|מז"א/i);
  });
});
