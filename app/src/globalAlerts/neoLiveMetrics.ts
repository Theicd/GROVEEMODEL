import type { GlobeAlertEvent } from "./types";
import type { NeoOrbitTrack, NeoTrackPoint } from "./neoTrack";

const LD_KM = 384_400;

/** Interpolate Horizons track to "now" for live globe + card metrics. */
export function interpolateNeoPoint(track: NeoOrbitTrack, now = Date.now()): NeoTrackPoint {
  const pts = track.points;
  if (!pts.length) return track.closest;
  if (now <= pts[0].t) return pts[0];
  const last = pts[pts.length - 1];
  if (now >= last.t) return last;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (now < a.t || now > b.t) continue;
    const span = b.t - a.t || 1;
    const f = (now - a.t) / span;
    return {
      t: now,
      lat: a.lat + (b.lat - a.lat) * f,
      lon: a.lon + (b.lon - a.lon) * f,
      distAu: a.distAu + (b.distAu - a.distAu) * f,
      distLd: a.distLd + (b.distLd - a.distLd) * f,
      deldotKmS: a.deldotKmS + (b.deldotKmS - a.deldotKmS) * f,
    };
  }
  return track.closest;
}

/** Estimate current geocentric distance (LD) from CAD v_rel until Horizons loads. */
export function estimateLiveDistLd(ev: GlobeAlertEvent, now = Date.now()): number {
  const ca = ev.approachTime ?? ev.time;
  const caLd = ev.distLd ?? 0;
  if (now >= ca) return caLd;
  const secLeft = Math.max(0, (ca - now) / 1000);
  const vKmS = ev.vRel ?? ev.vInf ?? 0;
  const extraLd = (vKmS * secLeft) / LD_KM;
  return Math.min(caLd + extraLd, caLd + 25);
}

export function estimateDiameterKm(ev: GlobeAlertEvent): number | undefined {
  if (ev.diameterKm != null && ev.diameterKm > 0) return ev.diameterKm;
  if (ev.hMagnitude == null) return undefined;
  return Math.pow(10, 3.1236 - 0.206 * ev.hMagnitude);
}

export function liveNeoMetrics(
  ev: GlobeAlertEvent,
  track?: NeoOrbitTrack | null,
  now = Date.now(),
): { distLd: number; speedKmS: number; diameterKm?: number } {
  const diameterKm = estimateDiameterKm(ev);
  if (ev.showcaseNeo) {
    return {
      distLd: ev.distLd ?? track?.closest.distLd ?? 10,
      speedKmS: ev.vRel ?? ev.vInf ?? 0,
      diameterKm,
    };
  }
  if (track && track.points.length >= 2) {
    const p = interpolateNeoPoint(track, now);
    return {
      distLd: p.distLd,
      speedKmS: Math.abs(p.deldotKmS) || ev.vRel || 0,
      diameterKm,
    };
  }
  return {
    distLd: estimateLiveDistLd(ev, now),
    speedKmS: ev.vRel ?? ev.vInf ?? 0,
    diameterKm,
  };
}
