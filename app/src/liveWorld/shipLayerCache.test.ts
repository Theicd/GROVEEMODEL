import { describe, expect, it } from "vitest";
import { mergeLiveShipItems, mergeShipLayers } from "./shipLayerCache";

describe("shipLayerCache", () => {
  it("mergeLiveShipItems prefers AISStream over Digitraffic at same coords", () => {
    const merged = mergeLiveShipItems(
      [{ name: "BALTIC", lat: 59.1, lon: 20.0, source: "ais" }],
      [{ name: "HAIFA STAR", lat: 32.82, lon: 35.0, source: "aisstream" }],
      [{ name: "BALTIC", lat: 59.1, lon: 20.0, source: "aisstream" }],
    );
    expect(merged).toHaveLength(2);
    expect(merged.find((s) => s.lat === 59.1)?.source).toBe("aisstream");
  });

  it("mergeShipLayers keeps AISStream when globe sends Digitraffic-only", () => {
    const existing = {
      regionLabel: "AISStream",
      count: 2,
      items: [
        { name: "MED ONE", lat: 33.0, lon: 34.0, source: "aisstream" },
        { name: "MED TWO", lat: 32.5, lon: 35.0, source: "aisstream" },
      ],
    };
    const incoming = {
      regionLabel: "עולם חי",
      count: 1,
      items: [{ name: "NORD", lat: 55.7, lon: 20.8, source: "digitraffic" }],
    };
    const merged = mergeShipLayers(existing, incoming);
    expect(merged?.items.some((s) => s.source === "aisstream")).toBe(true);
    expect(merged?.items.some((s) => s.name === "NORD")).toBe(true);
  });
});
