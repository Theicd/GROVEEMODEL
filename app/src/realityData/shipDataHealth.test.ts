import { describe, expect, it } from "vitest";

const HAIFA_BBOX = { minLat: 32.5, maxLat: 33.2, minLon: 34.6, maxLon: 35.2 };
const HELSINKI = { lat: 60.2, lon: 24.9, radius: 50 };

const countInBbox = (
  features: Array<{ geometry?: { coordinates?: [number, number] } }>,
  bbox: typeof HAIFA_BBOX,
): number =>
  features.filter((f) => {
    const c = f.geometry?.coordinates;
    if (!c) return false;
    const [lon, lat] = c;
    return lat >= bbox.minLat && lat <= bbox.maxLat && lon >= bbox.minLon && lon <= bbox.maxLon;
  }).length;

describe("shipDataHealth (Digitraffic live)", () => {
  it(
    "API is active and returns Baltic coverage (not Mediterranean)",
    async () => {
      const res = await fetch("https://meri.digitraffic.fi/api/ais/v1/locations", {
        signal: AbortSignal.timeout(20_000),
      });
      expect(res.ok).toBe(true);
      const geo = (await res.json()) as { features?: Array<{ geometry?: { coordinates?: [number, number] } }> };
      const features = geo.features ?? [];
      expect(features.length).toBeGreaterThan(1000);

      const haifaCount = countInBbox(features, HAIFA_BBOX);
      expect(haifaCount).toBe(0);

      const lats = features.map((f) => f.geometry?.coordinates?.[1]).filter((n): n is number => n != null);
      const lons = features.map((f) => f.geometry?.coordinates?.[0]).filter((n): n is number => n != null);
      expect(Math.min(...lats)).toBeGreaterThan(50);
      expect(Math.max(...lats)).toBeLessThan(67);
    },
    30_000,
  );

  it(
    "regional API returns vessels near Helsinki (positive control)",
    async () => {
      const url = `https://meri.digitraffic.fi/api/ais/v1/locations?latitude=${HELSINKI.lat}&longitude=${HELSINKI.lon}&radius=${HELSINKI.radius}`;
      const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
      expect(res.ok).toBe(true);
      const geo = (await res.json()) as { features?: unknown[] };
      expect((geo.features ?? []).length).toBeGreaterThan(10);
    },
    30_000,
  );
});
