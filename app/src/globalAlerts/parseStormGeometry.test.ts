import { describe, expect, it } from "vitest";
import { bearingToCompassHe, chainLineSegments, parseStormGeometry } from "./parseStormGeometry";

describe("parseStormGeometry", () => {
  it("chains line segments into observed path", () => {
    const track = parseStormGeometry([
      {
        geometry: { type: "LineString", coordinates: [[163, 9.2], [162.3, 9.8]] },
        properties: { Class: "TS Line_Line_0" },
      },
      {
        geometry: { type: "LineString", coordinates: [[162.3, 9.8], [161.4, 10.5]] },
        properties: { Class: "TS Line_Line_1" },
      },
    ]);
    expect(track.observed.length).toBe(3);
    expect(track.observed[0].lon).toBe(163);
    expect(track.observed[2].lat).toBe(10.5);
  });

  it("extracts forecast points from polygon centroids", () => {
    const track = parseStormGeometry([
      {
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [140, 16],
              [141, 16],
              [141, 17],
              [140, 17],
              [140, 16],
            ],
          ],
        },
        properties: { Class: "Point_Polygon_Point_0", polygonlabel: "01/07 18:00 UTC" },
      },
    ]);
    expect(track.forecast.length).toBe(1);
    expect(track.forecast[0].lat).toBeCloseTo(16.4, 0);
    expect(track.forecast[0].time).toBeTypeOf("number");
  });

  it("maps bearing to Hebrew compass", () => {
    expect(bearingToCompassHe(0)).toBe("צפון");
    expect(bearingToCompassHe(90)).toBe("מזרח");
  });
});

describe("chainLineSegments", () => {
  it("returns empty for no segments", () => {
    expect(chainLineSegments([])).toEqual([]);
  });
});
