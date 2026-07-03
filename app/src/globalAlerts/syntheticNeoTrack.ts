import type { GlobeAlertEvent } from "./types";
import type { NeoOrbitTrack, NeoTrackPoint } from "./neoTrack";

const AU_PER_LD = 384_400 / 149_597_870.7;

/** Fallback inbound ray when Horizons is unreachable — straight line toward Earth. */
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
  const farLd = Math.max(caLd + 8, caLd * 2.2);
  const steps = 18;
  const points: NeoTrackPoint[] = [];

  for (let i = 0; i <= steps; i++) {
    const frac = i / steps;
    const distLd = farLd - (farLd - caLd) * frac;
    const t = now + etaMs * frac;
    points.push({
      t,
      lat: caLat,
      lon: caLon,
      distAu: distLd * AU_PER_LD,
      distLd,
      deldotKmS: -vRel,
    });
  }

  const closest = points.reduce((a, b) => (a.distAu < b.distAu ? a : b));
  return { designation: des, points, closest };
}
