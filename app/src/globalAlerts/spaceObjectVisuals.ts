import * as THREE from "three";
import type { GlobeAlertEvent } from "./types";
import { estimateDiameterKm } from "./neoLiveMetrics";
import { MOON_SCENE_ANGULAR, visualRadiusFromLd } from "./neoTrack";

export type SpectralKey = "C" | "S" | "M" | "V" | "comet";

export type AsteroidShape =
  | "spherical"
  | "peanut"
  | "spinning_top"
  | "elongated"
  | "contact_binary"
  | "irregular";

export type SpectralProfile = {
  name: string;
  color: number;
  roughness: number;
  metalness: number;
  desc: string;
};

export const SPECTRAL_TYPES: Record<SpectralKey, SpectralProfile> = {
  C: {
    name: "פחמני (C-type)",
    color: 0x5c4a3a,
    roughness: 0.95,
    metalness: 0.05,
    desc: "עשיר בפחמן, כהה — הנפוץ ביותר (~75%)",
  },
  S: {
    name: "סיליקטי (S-type)",
    color: 0x9e9e8e,
    roughness: 0.85,
    metalness: 0.15,
    desc: "עשיר בסיליקטים, בהיר יחסית (~17%)",
  },
  M: {
    name: "מתכתי (M-type)",
    color: 0xb0b0c0,
    roughness: 0.6,
    metalness: 0.7,
    desc: "עשיר במתכות (ניקל-ברזל), מבריק",
  },
  V: {
    name: "בזלתי (V-type)",
    color: 0x8e8e7a,
    roughness: 0.8,
    metalness: 0.1,
    desc: "דומה לסלעים בזלתיים, נגזר מ-4 וסטה",
  },
  comet: {
    name: "שביט",
    color: 0x8899aa,
    roughness: 0.7,
    metalness: 0.1,
    desc: "גוף קרח המשחרר גז ואבק כשמתקרב לשמש",
  },
};

export const SHAPE_LABELS: Record<AsteroidShape, string> = {
  spherical: "כדורי",
  peanut: "בצורת בוטנה",
  spinning_top: "בצורת סביבון",
  elongated: "מוארך",
  contact_binary: "בינארי מגע",
  irregular: "אי-רגולרי",
};

export function isCometDesignation(designation?: string): boolean {
  if (!designation) return false;
  return /\d+P\//i.test(designation) || /\bC\//i.test(designation);
}

export function inferSpectralType(ev: GlobeAlertEvent): SpectralKey {
  if (ev.showcaseSpectral) return ev.showcaseSpectral;
  const name = ev.designation ?? ev.location ?? "";
  if (isCometDesignation(name)) return "comet";
  const h = ev.hMagnitude ?? 22;
  if (ev.isPha && h < 20) return "S";
  if (h < 16) return "M";
  if (h < 19) return "S";
  if (h > 23) return "C";
  return "S";
}

export function inferAsteroidShape(ev: GlobeAlertEvent, diameterKm?: number): AsteroidShape {
  if (ev.showcaseShape) return ev.showcaseShape;
  const d = diameterKm ?? estimateDiameterKm(ev) ?? 0.05;
  const name = ev.designation ?? ev.location ?? "";
  if (isCometDesignation(name)) {
    if (/67P|Churyumov/i.test(name)) return "contact_binary";
    return d > 3 ? "spherical" : "irregular";
  }
  if (d > 10) return "spherical";
  if (d > 2) return "peanut";
  if (d > 0.5) return "elongated";
  if (d > 0.15) return "spinning_top";
  return "irregular";
}

/**
 * World-space mesh radius scaled to shell distance so apparent size stays
 * a small fraction of the Moon — never larger on screen at the same distance.
 */
export function spaceDisplayMeshSize(baseSize: number, distLd: number): number {
  const shell = visualRadiusFromLd(distLd);
  const t = Math.min(1, Math.max(0, (baseSize - 0.028) / 0.14));
  const minFrac = 0.014;
  const maxFrac = 0.045 + t * 0.035;
  const angularR = MOON_SCENE_ANGULAR * (minFrac + (maxFrac - minFrac) * t);
  return shell * angularR;
}

/** Apparent angular radius / Moon angular radius — for tests and debug. */
export function asteroidAngularFraction(baseSize: number, distLd: number): number {
  const shell = visualRadiusFromLd(distLd);
  if (shell <= 0) return 0;
  return spaceDisplayMeshSize(baseSize, distLd) / shell / MOON_SCENE_ANGULAR;
}

export function neoVisualSize(diameterKm?: number): number {
  const dM = Math.max(10, (diameterKm ?? 0.05) * 1000);
  const logD = Math.log10(dM);
  const normalized = Math.max(0, Math.min(1, (logD - 1.5) / 4));
  return 0.028 + normalized * 0.16;
}

export function sizeComparisonLabel(diameterKm?: number): string {
  const d = diameterKm ?? 0;
  if (d >= 500) return "גודל עיר";
  if (d >= 50) return "גודל רבע עיר";
  if (d >= 5) return "גודל בניין";
  if (d >= 0.5) return "גודל אוטובוס";
  if (d >= 0.05) return "גודל מכונית";
  return "גודל חדר";
}

function distortGeometry(geo: THREE.BufferGeometry, amount: number, seed = 0) {
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const n = 1 + (Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453 % 1 - 0.5) * amount * 2;
    pos.setXYZ(i, x * n, y * n, z * n);
  }
  geo.computeVertexNormals();
}

export function createAsteroidGeometry(shape: AsteroidShape, baseSize: number, seed = 0): THREE.BufferGeometry {
  const sz = Math.max(0.01, baseSize);
  switch (shape) {
    case "peanut": {
      const g1 = new THREE.SphereGeometry(sz, 12, 10);
      const g2 = new THREE.SphereGeometry(sz * 0.75, 10, 8);
      g1.scale(1, 0.8, 0.7);
      g2.scale(1, 0.75, 0.65);
      const p1 = g1.attributes.position.array as Float32Array;
      const p2 = g2.attributes.position.array as Float32Array;
      const offset = sz * 1.2;
      const positions = new Float32Array(p1.length + p2.length);
      positions.set(p1, 0);
      for (let i = 0; i < p2.length; i += 3) {
        positions[p1.length + i] = p2[i] + offset;
        positions[p1.length + i + 1] = p2[i + 1];
        positions[p1.length + i + 2] = p2[i + 2];
      }
      g1.dispose();
      g2.dispose();
      const merged = new THREE.BufferGeometry();
      merged.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      merged.computeVertexNormals();
      distortGeometry(merged, 0.2, seed);
      return merged;
    }
    case "spinning_top": {
      const geo = new THREE.SphereGeometry(sz, 16, 12);
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const z = pos.getZ(i);
        const ny = y / sz;
        const bulge = 1 + 0.3 * Math.cos(ny * Math.PI);
        pos.setXYZ(i, x * bulge, y * 0.7, z * bulge);
      }
      geo.computeVertexNormals();
      distortGeometry(geo, 0.15, seed);
      return geo;
    }
    case "elongated": {
      const geo = new THREE.SphereGeometry(sz, 14, 10);
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        pos.setXYZ(i, pos.getX(i) * 2, pos.getY(i) * 0.7, pos.getZ(i) * 0.6);
      }
      geo.computeVertexNormals();
      distortGeometry(geo, 0.25, seed);
      return geo;
    }
    case "contact_binary": {
      const g1 = new THREE.SphereGeometry(sz, 12, 10);
      const g2 = new THREE.SphereGeometry(sz * 0.9, 12, 10);
      g1.scale(1.3, 0.8, 1);
      g2.scale(1, 0.9, 0.8);
      const p1 = g1.attributes.position.array as Float32Array;
      const p2 = g2.attributes.position.array as Float32Array;
      const offset = sz * 1.5;
      const positions = new Float32Array(p1.length + p2.length);
      positions.set(p1, 0);
      for (let i = 0; i < p2.length; i += 3) {
        positions[p1.length + i] = p2[i] + offset * 0.7;
        positions[p1.length + i + 1] = p2[i + 1] + offset * 0.4;
        positions[p1.length + i + 2] = p2[i + 2];
      }
      g1.dispose();
      g2.dispose();
      const merged = new THREE.BufferGeometry();
      merged.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      merged.computeVertexNormals();
      distortGeometry(merged, 0.2, seed);
      return merged;
    }
    case "spherical": {
      const geo = new THREE.SphereGeometry(sz, 16, 14);
      distortGeometry(geo, 0.08, seed);
      return geo;
    }
    case "irregular":
    default: {
      const geo = new THREE.IcosahedronGeometry(sz, 1);
      distortGeometry(geo, 0.35, seed);
      return geo;
    }
  }
}

export function buildNeoOrbitRing(pos: THREE.Vector3, eccVisual = 0.2): THREE.Vector3[] {
  const r = pos.length();
  const yOff = pos.y * 0.45;
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= 64; i++) {
    const t = (i / 64) * Math.PI * 2;
    const rr = r * (1 + eccVisual * 0.3 * Math.cos(t));
    pts.push(new THREE.Vector3(rr * Math.cos(t), yOff * Math.sin(t * 0.5), rr * Math.sin(t)));
  }
  return pts;
}

export function orbitLineColor(spectral: SpectralKey, isPha: boolean, liveLd: number): number {
  if (spectral === "comet") return 0x88bbdd;
  if (isPha || liveLd < 1) return 0xff4444;
  if (liveLd < 5) return 0xffcc44;
  return 0x44ff88;
}

/** Scene sun position — comet tails point away from here. */
export const SCENE_SUN_POSITION = new THREE.Vector3(140, 36, 60);

/** Short comet tail along local -X (oriented each frame away from the sun). */
export function createCometTail(parent: THREE.Object3D, meshRadius: number): THREE.Group {
  const tailGroup = new THREE.Group();
  const tailLen = Math.max(0.04, meshRadius * 2.2);
  const pointSize = Math.min(0.012, meshRadius * 0.14);

  const ionCount = 48;
  const ionPos = new Float32Array(ionCount * 3);
  const ionCol = new Float32Array(ionCount * 3);
  for (let i = 0; i < ionCount; i++) {
    const t = Math.random();
    const spread = t * meshRadius * 0.35;
    ionPos[i * 3] = -t * tailLen + (Math.random() - 0.5) * spread;
    ionPos[i * 3 + 1] = (Math.random() - 0.5) * spread;
    ionPos[i * 3 + 2] = (Math.random() - 0.5) * spread;
    ionCol[i * 3] = 0.3 + Math.random() * 0.2;
    ionCol[i * 3 + 1] = 0.5 + Math.random() * 0.3;
    ionCol[i * 3 + 2] = 0.9;
  }
  const ionGeo = new THREE.BufferGeometry();
  ionGeo.setAttribute("position", new THREE.BufferAttribute(ionPos, 3));
  ionGeo.setAttribute("color", new THREE.BufferAttribute(ionCol, 3));
  tailGroup.add(
    new THREE.Points(
      ionGeo,
      new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        transparent: true,
        opacity: 0.45,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    ),
  );

  const dustCount = 64;
  const dustPos = new Float32Array(dustCount * 3);
  const dustCol = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i++) {
    const t = Math.random();
    const spread = t * meshRadius * 0.45;
    dustPos[i * 3] = -t * tailLen * 0.85 + (Math.random() - 0.5) * spread;
    dustPos[i * 3 + 1] = (Math.random() - 0.5) * spread;
    dustPos[i * 3 + 2] = (Math.random() - 0.5) * spread;
    dustCol[i * 3] = 0.95;
    dustCol[i * 3 + 1] = 0.82;
    dustCol[i * 3 + 2] = 0.55;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  dustGeo.setAttribute("color", new THREE.BufferAttribute(dustCol, 3));
  tailGroup.add(
    new THREE.Points(
      dustGeo,
      new THREE.PointsMaterial({
        size: pointSize * 1.15,
        vertexColors: true,
        transparent: true,
        opacity: 0.28,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    ),
  );

  const coma = new THREE.Mesh(
    new THREE.SphereGeometry(meshRadius * 1.15, 8, 6),
    new THREE.MeshBasicMaterial({
      color: 0x88aacc,
      transparent: true,
      opacity: 0.08,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  tailGroup.add(coma);
  parent.add(tailGroup);
  return tailGroup;
}

const _tailAway = new THREE.Vector3();
const _tailX = new THREE.Vector3(-1, 0, 0);

/** Point tail anti-sunward from the comet nucleus world position. */
export function orientCometTail(tailGroup: THREE.Group, bodyWorldPos: THREE.Vector3): void {
  _tailAway.copy(bodyWorldPos).sub(SCENE_SUN_POSITION);
  if (_tailAway.lengthSq() < 1e-6) _tailAway.set(0, 0, 1);
  _tailAway.normalize();
  tailGroup.quaternion.setFromUnitVectors(_tailX, _tailAway);
}

/** Simple seeded PRNG for deterministic per-asteroid surfaces. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const _texCache = new Map<string, { map: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture }>();

/**
 * Procedural asteroid surface (albedo + bump) tinted per spectral type.
 * Cached by spectral+seed so repeated bodies reuse GPU textures.
 */
export function createAsteroidTexture(
  spectral: SpectralKey,
  seed: number,
): { map: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture } {
  const key = `${spectral}:${seed % 997}`;
  const cached = _texCache.get(key);
  if (cached) return cached;

  const W = 256;
  const H = 128;
  const base = SPECTRAL_TYPES[spectral].color;
  const br = (base >> 16) & 0xff;
  const bg = (base >> 8) & 0xff;
  const bb = base & 0xff;
  const rnd = mulberry32(seed + 1);

  const albedo = document.createElement("canvas");
  albedo.width = W;
  albedo.height = H;
  const ac = albedo.getContext("2d")!;
  const bump = document.createElement("canvas");
  bump.width = W;
  bump.height = H;
  const bc = bump.getContext("2d")!;

  ac.fillStyle = `rgb(${br},${bg},${bb})`;
  ac.fillRect(0, 0, W, H);
  bc.fillStyle = "rgb(128,128,128)";
  bc.fillRect(0, 0, W, H);

  // Mottled base variation.
  const img = ac.getImageData(0, 0, W, H);
  const bimg = bc.getImageData(0, 0, W, H);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (rnd() - 0.5) * 46;
    img.data[i] = Math.max(0, Math.min(255, br + n));
    img.data[i + 1] = Math.max(0, Math.min(255, bg + n));
    img.data[i + 2] = Math.max(0, Math.min(255, bb + n));
    const bn = 128 + (rnd() - 0.5) * 40;
    bimg.data[i] = bimg.data[i + 1] = bimg.data[i + 2] = bn;
  }
  ac.putImageData(img, 0, 0);
  bc.putImageData(bimg, 0, 0);

  // Craters — darker rim/floor on albedo, dents on bump.
  const craters = spectral === "comet" ? 8 : 22;
  for (let k = 0; k < craters; k++) {
    const cx = rnd() * W;
    const cy = rnd() * H;
    const r = 2 + rnd() * 14;
    const g = ac.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
    g.addColorStop(0, `rgba(0,0,0,${0.12 + rnd() * 0.18})`);
    g.addColorStop(0.7, "rgba(0,0,0,0.05)");
    g.addColorStop(1, `rgba(255,255,255,${0.05 + rnd() * 0.08})`);
    ac.fillStyle = g;
    ac.beginPath();
    ac.arc(cx, cy, r, 0, Math.PI * 2);
    ac.fill();

    const bg2 = bc.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
    bg2.addColorStop(0, "rgba(40,40,40,0.9)");
    bg2.addColorStop(1, "rgba(160,160,160,0.6)");
    bc.fillStyle = bg2;
    bc.beginPath();
    bc.arc(cx, cy, r, 0, Math.PI * 2);
    bc.fill();
  }

  const map = new THREE.CanvasTexture(albedo);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  const bumpMap = new THREE.CanvasTexture(bump);
  bumpMap.wrapS = bumpMap.wrapT = THREE.RepeatWrapping;

  const result = { map, bumpMap };
  _texCache.set(key, result);
  return result;
}

export type NeoVisualProfile = {
  spectral: SpectralKey;
  shape: AsteroidShape;
  visualSize: number;
  diameterKm?: number;
  rotSpeed: number;
  rotAxis: THREE.Vector3;
};

export function inferNeoVisualProfile(ev: GlobeAlertEvent, seed = 0): NeoVisualProfile {
  const diameterKm = estimateDiameterKm(ev);
  const spectral = inferSpectralType(ev);
  const shape = inferAsteroidShape(ev, diameterKm);
  const visualSize = neoVisualSize(diameterKm);
  const rotH = 4 + (seed % 20);
  const rotSpeed = (2 * Math.PI) / (rotH * 3600);
  const rotAxis = new THREE.Vector3(
    Math.sin(seed * 1.7) * 0.4,
    1,
    Math.cos(seed * 2.3) * 0.3,
  ).normalize();
  return { spectral, shape, visualSize, diameterKm, rotSpeed, rotAxis };
}

export function spectralHex(key: SpectralKey): string {
  return "#" + SPECTRAL_TYPES[key].color.toString(16).padStart(6, "0");
}
