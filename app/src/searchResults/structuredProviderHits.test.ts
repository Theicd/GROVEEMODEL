import { describe, expect, it } from "vitest";
import type { SearchSourceResult } from "../webSearch/types";
import { mergeSourcesToHits, buildUnifiedSearchPayload } from "./mergeSearchHits";
import { filterHits } from "./rankHits";
import {
  parsePlacesStructuredHits,
  parseRouteStructuredHits,
  parseWeatherStructuredHits,
} from "./structuredProviderHits";

describe("structuredProviderHits", () => {
  it("parses open-meteo weather lines", () => {
    const s: SearchSourceResult = {
      provider: "open-meteo",
      label: "מזג אוויר",
      ok: true,
      text: "מיקום: תל אביב\nטמפרטורה: 24°C\nמצב: מעונן\nתחזית שבועית:\n- 2026-06-16: 20–28°C",
      url: "https://open-meteo.com/",
      latencyMs: 1,
    };
    const hits = parseWeatherStructuredHits(s);
    expect(hits[0]?.kind).toBe("weather");
    expect(hits[0]?.title).toContain("24");
  });

  it("parses nominatim place with coords on next line", () => {
    const s: SearchSourceResult = {
      provider: "nominatim-places",
      label: "OSM",
      ok: true,
      text: "1. Flughafen BER · railway station\n   52.36399, 13.50833",
      url: "https://www.openstreetmap.org/",
      latencyMs: 1,
      geo: { lat: 52.36399, lon: 13.50833, label: "BER" },
    };
    const hits = parsePlacesStructuredHits(s);
    expect(hits[0]?.kind).toBe("place");
    expect(hits[0]?.snippet).toContain("52.36399");
  });

  it("parses osrm route", () => {
    const s: SearchSourceResult = {
      provider: "osrm-distance",
      label: "OSRM",
      ok: true,
      text: 'מ: BER\nאל: Flughafen Bahnhof\nמרחק נסיעה (OSRM): 2.1 ק"מ\nזמן נסיעה משוער: 5 דק\'',
      latencyMs: 1,
    };
    const hits = parseRouteStructuredHits(s);
    expect(hits[0]?.kind).toBe("route");
  });

  it("mergeSourcesToHits includes weather and place hits", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "open-meteo",
        label: "מזג אוויר",
        ok: true,
        text: "מיקום: חיפה\nטמפרטורה: 22°C\nמצב: בהיר",
        latencyMs: 1,
      },
      {
        provider: "nominatim-places",
        label: "OSM",
        ok: true,
        text: "1. Station\n   32.8, 34.9",
        latencyMs: 1,
      },
    ];
    const hits = mergeSourcesToHits(sources, "מזג אוויר בחיפה");
    expect(hits.some((h) => h.kind === "weather")).toBe(true);
  });

  it("filterHits weather tab still works; events tab includes weather and open-meteo marine", () => {
    const hits = [
      { id: "1", kind: "weather" as const, title: "w", url: "u", snippet: "s", sourceLabel: "l", provider: "open-meteo" as const, summarizable: false },
      { id: "2", kind: "marine" as const, title: "m", url: "u", snippet: "s", sourceLabel: "l", provider: "osm-overpass-marine" as const, summarizable: false },
    ];
    expect(filterHits(hits, "weather")).toHaveLength(1);
    expect(filterHits(hits, "ships")).toHaveLength(1);
    expect(filterHits(hits, "events")).toHaveLength(1);
  });

  it("buildUnifiedSearchPayload prefers events tab for weather queries", () => {
    const payload = buildUnifiedSearchPayload("מה מזג האוויר בתל אביב", [
      {
        provider: "open-meteo",
        label: "מזג אוויר",
        ok: true,
        text: "מיקום: תל אביב\nטמפרטורה: 25°C\nמצב: בהיר",
        latencyMs: 1,
      },
    ]);
    expect(payload.preferEventsFilter).toBe(true);
    expect(payload.facets.weather).toBeGreaterThan(0);
  });
});
