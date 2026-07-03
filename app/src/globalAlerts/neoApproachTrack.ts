import { interpolateNeoPoint } from "./neoLiveMetrics";
import type { NeoOrbitTrack, NeoTrackPoint } from "./neoTrack";

const AU_PER_LD = 384_400 / 149_597_870.7;

/** Straight inbound segment for 3D display — avoids orbital U-turn artifacts. */
export function buildApproachDisplayTrack(track: NeoOrbitTrack, now = Date.now()): NeoOrbitTrack {
  const closest = track.closest;
  const current = interpolateNeoPoint(track, now);
  const startLd = Math.max(current.distLd, closest.distLd * 1.02);
  const steps = 18;
  const points: NeoTrackPoint[] = [];

  for (let i = 0; i <= steps; i++) {
    const f = i / steps;
    const distLd = startLd + (closest.distLd - startLd) * f;
    const lat = current.lat + (closest.lat - current.lat) * f;
    const lon = current.lon + (closest.lon - current.lon) * f;
    const span = Math.max(closest.t - now, 120_000);
    points.push({
      t: now + span * f,
      lat,
      lon,
      distAu: distLd * AU_PER_LD,
      distLd,
      deldotKmS: current.deldotKmS,
    });
  }

  return { designation: track.designation, points, closest, bearingDeg: track.bearingDeg };
}

export function approachClosurePercent(
  liveLd: number,
  caLd: number,
  farLd: number,
): number {
  const span = Math.max(farLd - caLd, 0.05);
  const pct = 100 - ((liveLd - caLd) / span) * 100;
  return Math.max(2, Math.min(98, pct));
}
