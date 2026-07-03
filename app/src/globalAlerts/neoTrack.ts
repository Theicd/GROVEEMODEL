import * as THREE from "three";
import { latLonToVec3 } from "./alignToSurface";

export type NeoTrackPoint = {
  t: number;
  lat: number;
  lon: number;
  distAu: number;
  distLd: number;
  deldotKmS: number;
};

export type NeoOrbitTrack = {
  designation: string;
  points: NeoTrackPoint[];
  closest: NeoTrackPoint;
  bearingDeg?: number;
};

const AU_PER_LD = 384_400 / 149_597_870.7;

/** Map lunar distance → scene radius (Earth surface = 1). Spread NEOs visibly in space. */
export function visualRadiusFromLd(distLd: number): number {
  const ld = Math.max(0.05, distLd);
  if (ld <= 1) return 1.1 + ld * 0.18;
  if (ld <= 6) return 1.28 + (ld - 1) * 0.2;
  if (ld <= 15) return 2.28 + (ld - 6) * 0.12;
  return Math.min(3.8, 3.36 + (ld - 15) * 0.04);
}

export function visualRadiusFromDistAu(distAu: number): number {
  return visualRadiusFromLd(distAu / AU_PER_LD);
}

export function neoSpacePosition(lat: number, lon: number, distLd: number): THREE.Vector3 {
  const dir = latLonToVec3(lat, lon, 1).normalize();
  return dir.multiplyScalar(visualRadiusFromLd(distLd));
}

export function geoDirectionToVisualVec(lat: number, lon: number, distAu: number): THREE.Vector3 {
  return neoSpacePosition(lat, lon, distAu / AU_PER_LD);
}

/** ICRF equatorial RA/Dec + geocentric range → lat/lon anchor. */
export function raDecDistToLatLon(raHours: number, decDeg: number): { lat: number; lon: number } {
  const ra = (raHours * 15 * Math.PI) / 180;
  const dec = (decDeg * Math.PI) / 180;
  const x = Math.cos(dec) * Math.cos(ra);
  const y = Math.sin(dec);
  const z = Math.cos(dec) * Math.sin(ra);
  const r = Math.sqrt(x * x + y * y + z * z) || 1;
  const nx = x / r;
  const ny = y / r;
  const nz = z / r;
  const phi = Math.acos(Math.max(-1, Math.min(1, ny)));
  const lat = 90 - (phi * 180) / Math.PI;
  const theta = Math.atan2(nz, -nx);
  const lon = (theta * 180) / Math.PI - 180;
  return { lat, lon };
}

export function buildNeoTrackFromHorizons(
  designation: string,
  rows: Array<{
    timeLabel: string;
    raHours: number;
    decDeg: number;
    distAu: number;
    deldotKmS: number;
    t?: number;
  }>,
): NeoOrbitTrack | null {
  if (!rows.length) return null;

  const points: NeoTrackPoint[] = rows.map((r) => {
    const { lat, lon } = raDecDistToLatLon(r.raHours, r.decDeg);
    return {
      t: r.t ?? (Date.parse(r.timeLabel.replace(" ", "T") + ":00Z") || Date.now()),
      lat,
      lon,
      distAu: r.distAu,
      distLd: r.distAu / AU_PER_LD,
      deldotKmS: r.deldotKmS,
    };
  });

  const closest = points.reduce((a, b) => (a.distAu < b.distAu ? a : b));

  let bearingDeg: number | undefined;
  if (points.length >= 2) {
    const i = points.indexOf(closest);
    const from = points[Math.max(0, i - 1)];
    const to = points[Math.min(points.length - 1, i + 1)];
    const a = latLonToVec3(from.lat, from.lon, 1);
    const b = latLonToVec3(to.lat, to.lon, 1);
    const d = b.clone().sub(a);
    if (d.lengthSq() > 1e-8) {
      const east = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), a).normalize();
      const north = new THREE.Vector3().crossVectors(a, east).normalize();
      const de = d.dot(east);
      const dn = d.dot(north);
      bearingDeg = ((Math.atan2(de, dn) * 180) / Math.PI + 360) % 360;
    }
  }

  return { designation, points, closest, bearingDeg };
}
