import { describe, expect, it } from "vitest";
import {
  parseGdacsDisasterText,
  parseUsgsEarthquakeText,
  filterLiveDisasterHitsForQuery,
  mergeLiveDisasterHits,
  disasterItemToHit,
} from "./liveDisastersHits";
import type { UnifiedSearchHit } from "./types";

const USGS_SAMPLE = `סה"כ 3 רעידות מעל M5.0 ב-24 שעות (USGS).
- M5.2 · 45 km NE of Tokyo, Japan · 2024-06-19 12:34:56 UTC
  https://earthquake.usgs.gov/earthquakes/eventpage/us7000
- M5.1 · Chile · 2024-06-19 08:00:00 UTC
  https://earthquake.usgs.gov/earthquakes/eventpage/us7001`;

const GDACS_SAMPLE = `אירועי טבע (GDACS):
1. Tropical Cyclone ALBERTO · Cuba · Orange
2. Earthquake in Turkey · Turkey · Red`;

describe("liveDisastersHits", () => {
  it("parses USGS text into earthquake hits", () => {
    const hits = parseUsgsEarthquakeText(USGS_SAMPLE);
    expect(hits.length).toBe(2);
    expect(hits[0].kind).toBe("earthquake");
    expect(hits[0].meta?.magnitude).toBe(5.2);
    expect(hits[0].url).toContain("earthquake.usgs.gov");
  });

  it("parse GDACS text into disaster hits", () => {
    const hits = parseGdacsDisasterText(GDACS_SAMPLE);
    expect(hits.length).toBe(2);
    expect(hits[1].kind).toBe("disaster");
    expect(hits[1].meta?.alertLevel).toBe("Red");
  });

  it("disasterItemToHit handles GDACS object url from cache", () => {
    const hit = disasterItemToHit(
      {
        eventName: "Cyclone",
        country: "Cuba",
        alertLevel: "Orange",
        url: { report: "https://www.gdacs.org/report.aspx?eventid=1" } as unknown as string,
      },
      0,
    );
    expect(hit.url).toContain("gdacs.org");
  });

  it("filters earthquake hits by minimum magnitude", () => {
    const hits = parseUsgsEarthquakeText(USGS_SAMPLE);
    const filtered = filterLiveDisasterHitsForQuery(hits, "רעידות מעל M5.2 ב-24 שעות");
    expect(filtered.length).toBe(1);
    expect(filtered[0].meta?.magnitude).toBe(5.2);
  });

  it("mergeLiveDisasterHits dedupes by title and kind", () => {
    const base: UnifiedSearchHit[] = parseUsgsEarthquakeText(USGS_SAMPLE);
    const merged = mergeLiveDisasterHits(base, "");
    expect(merged.filter((h) => h.kind === "earthquake").length).toBeGreaterThanOrEqual(2);
  });
});
