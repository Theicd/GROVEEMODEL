import { describe, expect, it, vi } from "vitest";
import { detectRegionPreset } from "../realityData/shipRegion";
import { portsInBbox, MED_PORTS } from "../realityData/medPorts";
import { HAIFA_BAY_BBOX } from "../realityData/shipRegion";

describe("shipRegion", () => {
  it("detects Haifa bay from Hebrew query", () => {
    expect(detectRegionPreset("כמה כלי שייט במפרץ חיפה?")).toBe("haifa");
  });

  it("detects Suez canal", () => {
    expect(detectRegionPreset("אוניות בתעלת סואץ")).toBe("suez");
  });

  it("detects Rotterdam from Hebrew query", () => {
    expect(detectRegionPreset("אוניות ליד רוטרדם")).toBe("rotterdam");
  });

  it("filters two Haifa route markers inside Haifa bbox", () => {
    const inBay = portsInBbox(MED_PORTS, HAIFA_BAY_BBOX);
    expect(inBay.filter((p) => /Haifa/i.test(p.name)).length).toBeGreaterThanOrEqual(2);
  });
});

describe("ships intents", () => {
  it("classifies ships vs marine-infra separately", async () => {
    const { classifySearchIntents, isMarineInfraQuery, isShipsQuery } = await import("../webSearch/intents");
    expect(classifySearchIntents("כמה כלי שייט במפרץ חיפה?")).toContain("ships");
    expect(classifySearchIntents("כמה כלי שייט במפרץ חיפה?")).not.toContain("marine-infra");
    expect(classifySearchIntents("כמה מצופים במפרץ חיפה?")).toContain("marine-infra");
    expect(isMarineInfraQuery("כמה מצופים במפרץ חיפה?")).toBe(true);
    expect(isShipsQuery("כמה מצופים במפרץ חיפה?")).toBe(false);
  });
});

describe("resolveShipRegion", () => {
  it("uses country bbox for Greece lighthouse query", async () => {
    const { resolveShipRegion } = await import("../realityData/shipRegion");
    const region = await resolveShipRegion("מגדלורים ליד יוון");
    expect(region.label).toMatch(/יוון/);
    expect(region.bbox?.maxLat).toBeGreaterThan(40);
  });
});

describe("fetchShipsSearch", () => {
  it("returns route markers for Haifa when AIS empty", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("digitraffic")) {
          return new Response(JSON.stringify({ features: [] }), { status: 200 });
        }
        throw new Error(`unexpected fetch ${url}`);
      }),
    );
    const { fetchShipsSearch } = await import("../realityData/providers/ships");
    const result = await fetchShipsSearch("כמה כלי שייט במפרץ חיפה?");
    expect(result.ok).toBe(true);
    expect(result.text).toMatch(/ANSWER \(ships live\): 0/);
    expect(result.text).toMatch(/סימוני מסלול/);
    expect(result.text).toMatch(/מפרץ חיפה/);
    vi.unstubAllGlobals();
  });

  it("returns honest 0 live for Suez with demo markers separate", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("digitraffic")) {
          return new Response(JSON.stringify({ features: [] }), { status: 200 });
        }
        throw new Error(`unexpected fetch ${url}`);
      }),
    );
    const { fetchShipsSearch } = await import("../realityData/providers/ships");
    const result = await fetchShipsSearch("כמה אוניות נמצאות כרגע בתעלת סואץ?");
    expect(result.ok).toBe(true);
    expect(result.text).toMatch(/ANSWER \(ships live\): 0/);
    expect(result.text).toMatch(/תעלת סואץ/);
    expect(result.text).toMatch(/סימוני מסלול \(הדגמה/);
    vi.unstubAllGlobals();
  });

  it("resolves Rotterdam with AIS + route marker", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("digitraffic")) {
          return new Response(
            JSON.stringify({
              features: [
                {
                  geometry: { coordinates: [4.5, 51.93] },
                  properties: { mmsi: 123, sog: 100 },
                },
              ],
            }),
            { status: 200 },
          );
        }
        throw new Error(`unexpected fetch ${url}`);
      }),
    );
    const { fetchShipsSearch } = await import("../realityData/providers/ships");
    const result = await fetchShipsSearch("כמה אוניות ליד רוטרדם?");
    expect(result.ok).toBe(true);
    expect(result.text).toMatch(/רוטרדם/);
    expect(result.text).toMatch(/ANSWER \(ships live\): 1/);
    vi.unstubAllGlobals();
  });
});
