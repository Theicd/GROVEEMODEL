export type StormTrackPoint = {
  lat: number;
  lon: number;
  label?: string;
  kind: "observed" | "forecast";
  time?: number;
};

export type StormTrack = {
  observed: StormTrackPoint[];
  forecast: StormTrackPoint[];
  /** Movement bearing in degrees (0 = north, 90 = east). */
  bearingDeg?: number;
  /** Approximate speed in km/h from last two forecast steps. */
  speedKmh?: number;
};

type GeoFeature = {
  geometry?: {
    type?: string;
    coordinates?: unknown;
  };
  properties?: {
    polygonlabel?: string;
    Class?: string;
  };
};

const EPS = 0.05;

const near = (a: { lat: number; lon: number }, b: { lat: number; lon: number }) =>
  Math.abs(a.lat - b.lat) < EPS && Math.abs(a.lon - b.lon) < EPS;

const parseForecastTime = (label: string): number | undefined => {
  const m = label.match(/(\d{2})\/(\d{2})\s+(\d{2}):(\d{2})/);
  if (!m) return undefined;
  const year = new Date().getUTCFullYear();
  return Date.UTC(year, Number(m[2]) - 1, Number(m[1]), Number(m[3]), Number(m[4]));
};

const polygonCentroid = (coords: unknown): { lat: number; lon: number } | null => {
  if (!Array.isArray(coords) || !Array.isArray(coords[0])) return null;
  const ring = coords[0] as number[][];
  if (!ring.length) return null;
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const pt of ring) {
    if (!Array.isArray(pt) || pt.length < 2) continue;
    sx += Number(pt[0]);
    sy += Number(pt[1]);
    n++;
  }
  if (!n) return null;
  return { lon: sx / n, lat: sy / n };
};

/** Chain 2-point LineString segments into one ordered path. */
export function chainLineSegments(
  segments: Array<{ a: { lat: number; lon: number }; b: { lat: number; lon: number } }>,
): StormTrackPoint[] {
  if (!segments.length) return [];
  const remaining = [...segments];
  const out: StormTrackPoint[] = [{ ...remaining[0].a, kind: "observed" }];
  let cur = remaining[0].a;
  let end = remaining[0].b;
  remaining.shift();
  out.push({ ...end, kind: "observed" });
  cur = end;

  while (remaining.length) {
    const idx = remaining.findIndex((s) => near(s.a, cur) || near(s.b, cur));
    if (idx < 0) break;
    const seg = remaining.splice(idx, 1)[0];
    const next = near(seg.a, cur) ? seg.b : seg.a;
    if (!near(next, cur)) out.push({ ...next, kind: "observed" });
    cur = next;
  }

  return out;
}

export function bearingDeg(
  from: { lat: number; lon: number },
  to: { lat: number; lon: number },
): number {
  const φ1 = (from.lat * Math.PI) / 180;
  const φ2 = (to.lat * Math.PI) / 180;
  const Δλ = ((to.lon - from.lon) * Math.PI) / 180;
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

export function bearingToCompassHe(deg: number): string {
  const labels = ["צפון", "צ-מז", "מזרח", "ד-מז", "דרום", "ד-מע", "מערב", "צ-מע"];
  const idx = Math.round(deg / 45) % 8;
  return labels[idx];
}

const haversineKm = (
  a: { lat: number; lon: number },
  b: { lat: number; lon: number },
): number => {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const s1 = Math.sin(dLat / 2);
  const s2 = Math.sin(dLon / 2);
  const h = s1 * s1 + Math.cos((a.lat * Math.PI) / 180) * Math.cos((b.lat * Math.PI) / 180) * s2 * s2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
};

export function parseStormGeometry(features: GeoFeature[]): StormTrack {
  const lineSegs: Array<{ a: { lat: number; lon: number }; b: { lat: number; lon: number } }> = [];
  const forecast: StormTrackPoint[] = [];

  for (const f of features) {
    const cls = f.properties?.Class ?? "";
    const label = f.properties?.polygonlabel ?? "";
    const geom = f.geometry;

    if (geom?.type === "LineString" && Array.isArray(geom.coordinates)) {
      const c = geom.coordinates as number[][];
      if (c.length >= 2) {
        lineSegs.push({
          a: { lon: c[0][0], lat: c[0][1] },
          b: { lon: c[c.length - 1][0], lat: c[c.length - 1][1] },
        });
      }
      continue;
    }

    if (/Point_Polygon_Point/i.test(cls) && geom?.type === "Polygon") {
      const cen = polygonCentroid(geom.coordinates);
      if (!cen) continue;
      forecast.push({
        lat: cen.lat,
        lon: cen.lon,
        label,
        kind: "forecast",
        time: parseForecastTime(label),
      });
    }
  }

  forecast.sort((a, b) => (a.time ?? 0) - (b.time ?? 0));

  const observed = chainLineSegments(lineSegs);

  let bearing: number | undefined;
  let speedKmh: number | undefined;

  const moveFrom = observed.length > 1 ? observed[observed.length - 2] : observed[observed.length - 1];
  const moveTo = forecast[0] ?? observed[observed.length - 1];
  if (moveFrom && moveTo && !near(moveFrom, moveTo)) {
    bearing = bearingDeg(moveFrom, moveTo);
  }

  if (forecast.length >= 2) {
    const a = forecast[forecast.length - 2];
    const b = forecast[forecast.length - 1];
    const dist = haversineKm(a, b);
    const dtH = a.time && b.time ? (b.time - a.time) / 3600000 : 6;
    if (dtH > 0) speedKmh = dist / dtH;
  }

  return { observed, forecast, bearingDeg: bearing, speedKmh };
}

export function stormPositionNow(track: StormTrack, now = Date.now()): { lat: number; lon: number } {
  const pts = track.forecast.filter((p) => p.time != null);
  if (pts.length >= 2) {
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      if (a.time! <= now && now <= b.time!) {
        const t = (now - a.time!) / (b.time! - a.time!);
        return { lat: a.lat + (b.lat - a.lat) * t, lon: a.lon + (b.lon - a.lon) * t };
      }
    }
    if (now >= pts[pts.length - 1].time!) return pts[pts.length - 1];
  }
  if (track.observed.length) return track.observed[track.observed.length - 1];
  if (track.forecast.length) return track.forecast[0];
  return { lat: 0, lon: 0 };
}
