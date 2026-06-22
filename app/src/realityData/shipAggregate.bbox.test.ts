import { describe, expect, it, vi, beforeEach } from "vitest";
import { HAIFA_BAY_BBOX } from "./shipRegion";

vi.mock("../apiKeys/apiKeyStore", () => ({
  isAisStreamConfigured: () => true,
}));

const fetchAisStreamShips = vi.fn();
const fetchAisStreamGlobeShips = vi.fn();

vi.mock("./providers/aisStream", () => ({
  fetchAisStreamShips: (...args: unknown[]) => fetchAisStreamShips(...args),
  fetchAisStreamGlobeShips: (...args: unknown[]) => fetchAisStreamGlobeShips(...args),
}));

vi.mock("../webSearch/fetchJson", () => ({
  fetchJson: vi.fn(async () => []),
}));

vi.mock("../liveWorld/snapshotStore", () => ({
  getCachedLiveWorldSnapshot: () => null,
}));

describe("aggregateShipHits regional bbox", () => {
  beforeEach(() => {
    fetchAisStreamShips.mockReset();
    fetchAisStreamGlobeShips.mockReset();
  });

  it("does not count global AISStream ships for Haifa bay query", async () => {
    fetchAisStreamShips.mockResolvedValue([]);
    fetchAisStreamGlobeShips.mockResolvedValue([
      { name: "Hamburg", lat: 53.54, lon: 9.88, source: "aisstream" as const },
      { name: "NYC", lat: 40.72, lon: -73.97, source: "aisstream" as const },
      { name: "Haifa Star", lat: 32.82, lon: 35.0, source: "aisstream" as const },
    ]);

    const { aggregateShipHits, formatShipsText } = await import("./shipAggregate");
    const region = {
      label: "מפרץ חיפה",
      bbox: HAIFA_BAY_BBOX,
      center: { lat: 32.82, lon: 35.0 },
      radiusNm: 60,
    };

    const agg = await aggregateShipHits("כמה אוניות נמצאות כרגע במפרץ חיפה?", region);
    expect(agg.liveHits).toHaveLength(0);
    expect(fetchAisStreamGlobeShips).not.toHaveBeenCalled();

    fetchAisStreamShips.mockResolvedValue([
      { name: "Haifa Star", lat: 32.82, lon: 35.0, source: "aisstream" as const },
    ]);
    const agg2 = await aggregateShipHits("כמה אוניות במפרץ חיפה?", region);
    expect(agg2.liveHits).toHaveLength(1);
    expect(agg2.liveHits[0].name).toBe("Haifa Star");

    const text = formatShipsText(region.label, agg2, "כמה אוניות במפרץ חיפה?");
    expect(text).toMatch(/ANSWER \(ships live\): 1/);
    expect(text).not.toMatch(/53\.54,9\.88/);
  });
});
