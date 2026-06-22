import { describe, expect, it, beforeEach } from "vitest";
import { parseGdacsFeatures } from "../realityData/providers/disasters";
import {
  buildLiveDisastersPayload,
  buildLiveDisastersHitsFromSnapshot,
  disasterItemToHit,
} from "./liveDisastersHits";
import { createEmptySearchPayload } from "./panelSearch";
import { clearLiveWorldSnapshotCache, setLiveWorldSnapshot } from "../liveWorld/snapshotStore";

describe("Search panel open QA", () => {
  beforeEach(() => {
    clearLiveWorldSnapshotCache();
  });

  it("createEmptySearchPayload does not throw with empty cache", () => {
    expect(() => createEmptySearchPayload()).not.toThrow();
    const payload = createEmptySearchPayload();
    expect(payload.query).toBe("");
    expect(Array.isArray(payload.hits)).toBe(true);
    expect(payload.facets).toBeDefined();
  });

  it("createEmptySearchPayload loads live events without auto-tab on empty open", () => {
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "fetch",
      earthquake: {
        feedLabel: "USGS",
        items: [
          {
            magnitude: 5.4,
            place: "Japan",
            time: Date.now(),
            url: "https://earthquake.usgs.gov/earthquakes/eventpage/us7000",
          },
        ],
      },
      disasters: {
        feedLabel: "GDACS",
        fetchedAt: Date.now(),
        items: [
          {
            eventName: "Earthquake in Turkey",
            country: "Turkey",
            alertLevel: "Orange",
            url: { report: "https://www.gdacs.org/report.aspx?eventid=999" },
          },
        ],
      },
    });

    const payload = createEmptySearchPayload();
    expect(payload.hits.length).toBeGreaterThanOrEqual(2);
    expect(payload.preferEventsFilter).toBe(false);
    expect(payload.facets.earthquakes).toBe(1);
    expect(payload.facets.disasters).toBe(1);
    expect(payload.liveDisastersNote).toMatch(/USGS/);
    expect(payload.hits.find((h) => h.kind === "disaster")?.url).toContain("gdacs.org");
  });

  it("createEmptySearchPayload loads GDACS without auto-tab on empty open", () => {
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "fetch",
      disasters: {
        feedLabel: "GDACS",
        fetchedAt: Date.now(),
        items: [
          {
            eventName: "Storm",
            country: "Cuba",
            alertLevel: "Orange",
            url: "https://www.gdacs.org/x",
          },
        ],
      },
    });
    const payload = createEmptySearchPayload();
    expect(payload.preferEventsFilter).toBe(false);
    expect(payload.facets.disasters).toBe(1);
  });

  it("createEmptySearchPayload includes ships in all tab when cache has AISStream vessels", () => {
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "globe",
      ships: {
        regionLabel: "AISStream",
        count: 2,
        items: [
          { name: "HAIFA STAR", lat: 32.82, lon: 35.0, source: "aisstream", speedKn: 8.2 },
          { name: "MED CARGO", lat: 33.1, lon: 34.5, source: "aisstream" },
        ],
      },
    });
    const payload = createEmptySearchPayload();
    expect(payload.facets.ships).toBe(2);
    expect(payload.preferShipsFilter).toBeUndefined();
    expect(payload.hits.filter((h) => h.kind === "ship")).toHaveLength(2);
  });

  it("parseGdacsFeatures normalizes object url from API", () => {
    const items = parseGdacsFeatures([
      {
        properties: {
          eventname: "Tropical Cyclone ALBERTO",
          country: "Cuba",
          alertlevel: "Orange",
          url: { report: "https://www.gdacs.org/report.aspx?eventid=555" },
        },
      },
    ]);
    expect(items[0].url).toBe("https://www.gdacs.org/report.aspx?eventid=555");
  });

  it("disasterItemToHit survives malformed url without throwing", () => {
    const hit = disasterItemToHit(
      {
        eventName: "Test Event",
        country: "Test",
        alertLevel: "Red",
        url: { nested: { href: "https://www.gdacs.org/x" } } as unknown as string,
      },
      0,
    );
    expect(hit.url).toBe("https://www.gdacs.org/x");
    expect(hit.kind).toBe("disaster");
  });

  it("buildLiveDisastersHitsFromSnapshot skips only broken rows", () => {
    const hits = buildLiveDisastersHitsFromSnapshot({
      fetchedAt: Date.now(),
      source: "mixed",
      disasters: {
        feedLabel: "GDACS",
        fetchedAt: Date.now(),
        items: [
          {
            eventName: "Good",
            country: "A",
            alertLevel: "Green",
            url: "https://www.gdacs.org/a",
          },
          {
            eventName: "Also Good",
            country: "B",
            alertLevel: "Orange",
            url: { url: "https://www.gdacs.org/b" } as unknown as string,
          },
        ],
      },
    });
    expect(hits).toHaveLength(2);
  });

  it("buildLiveDisastersPayload returns valid payload shape", () => {
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "fetch",
      earthquake: {
        feedLabel: "USGS",
        items: [{ magnitude: 4.2, place: "Chile", time: Date.now() }],
      },
    });
    const payload = buildLiveDisastersPayload("");
    expect(payload.generatedAt).toBeGreaterThan(0);
    expect(payload.providerErrors).toEqual([]);
    expect(payload.hits.some((h) => h.kind === "earthquake")).toBe(true);
  });
});
