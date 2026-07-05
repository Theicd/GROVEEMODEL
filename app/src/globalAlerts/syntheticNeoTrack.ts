import type { GlobeAlertEvent } from "./types";
import type { NeoOrbitTrack, NeoTrackPoint } from "./neoTrack";

const AU_PER_LD = 384_400 / 149_597_870.7;

function hashUnit(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967296;
}

/** Fallback inbound arc when Horizons is unreachable. */
export function buildSyntheticNeoTrack(ev: GlobeAlertEvent): NeoOrbitTrack {
  const des = ev.designation ?? ev.location;
  const caLat = ev.lat;
  const caLon = ev.lon;
  const caAu = ev.distAu ?? (ev.distLd ?? 1) * AU_PER_LD;
  const caLd = caAu / AU_PER_LD;
  const vRel = ev.vRel ?? 8;
  const approachTime = ev.approachTime ?? ev.time;
  const now = Date.now();
  const etaMs = Math.max(approachTime - now, 3_600_000);
  const farLd = Math.max(caLd + 12, caLd * 2.8);
  const steps = 32;
  const points: NeoTrackPoint[] = [];

  // Per-object inbound bearing so trajectories fan out instead of stacking.
  const bearing = hashUnit(des) * Math.PI * 2;
  const spreadDeg = 45 + hashUnit(des + "s") * 35;
  const lonStart = caLon + Math.cos(bearing) * spreadDeg;
  const latStart = caLat + Math.sin(bearing) * spreadDeg * 0.6;

  for (let i = 0; i <= steps; i++) {
    const frac = i / steps;
    const distLd = farLd - (farLd - caLd) * frac;
    const arc = Math.sin(frac * Math.PI);
    const lat = caLat + (latStart - caLat) * (1 - frac) * arc + Math.sin(frac * Math.PI * 2) * 4;
    const lon = caLon + (lonStart - caLon) * (1 - frac) * arc;
    const t = now + etaMs * frac;
    points.push({
      t,
      lat,
      lon,
      distAu: distLd * AU_PER_LD,
      distLd,
      deldotKmS: -vRel,
    });
  }

  const closest = points.reduce((a, b) => (a.distAu < b.distAu ? a : b));
  return { designation: des, points, closest };
}
