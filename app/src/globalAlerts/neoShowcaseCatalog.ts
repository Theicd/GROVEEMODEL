import type { NeoOrbitTrack, NeoTrackPoint } from "./neoTrack";
import type { AsteroidShape, SpectralKey } from "./spaceObjectVisuals";
import type { GlobeAlertEvent } from "./types";

export type ShowcaseEntry = {
  id: string;
  designation: string;
  nameHe: string;
  spectral: SpectralKey;
  shape: AsteroidShape;
  diamM: number;
  distLd: number;
  distSunAu: number;
  vRel: number;
  rotH: number;
  ecc: number;
  inc: number;
  orbitPhase: number;
  discovery: string;
  isPha?: boolean;
  desc: string;
};

/** Famous periodic NEOs / comets — orbit visually, no trail lines. */
export const NEO_SHOWCASE: ShowcaseEntry[] = [
  {
    id: "showcase-bennu",
    designation: "101955",
    nameHe: "101955 בנו",
    spectral: "C",
    shape: "peanut",
    diamM: 490,
    distLd: 8.2,
    distSunAu: 1.126,
    vRel: 28,
    rotH: 4.3,
    ecc: 0.204,
    inc: 6,
    orbitPhase: 0.2,
    discovery: "1999",
    isPha: true,
    desc: 'יעד משימת OSIRIS-REx. צורת אגוז מעופף.',
  },
  {
    id: "showcase-apophis",
    designation: "99942",
    nameHe: "99942 אפופיס",
    spectral: "S",
    shape: "elongated",
    diamM: 370,
    distLd: 4.5,
    distSunAu: 0.922,
    vRel: 30.7,
    rotH: 30.4,
    ecc: 0.191,
    inc: 3.3,
    orbitPhase: 1.1,
    discovery: "2004",
    isPha: true,
    desc: "אסטרואיד אטני. צפוי להתקרב ב-2029.",
  },
  {
    id: "showcase-ryugu",
    designation: "162173",
    nameHe: "162173 ריוגו",
    spectral: "C",
    shape: "spinning_top",
    diamM: 896,
    distLd: 6.1,
    distSunAu: 1.19,
    vRel: 28.3,
    rotH: 7.6,
    ecc: 0.19,
    inc: 5.9,
    orbitPhase: 2.4,
    discovery: "1999",
    desc: "יעד HAYABUSA2. צורת סביבון.",
  },
  {
    id: "showcase-itokawa",
    designation: "25143",
    nameHe: "25143 איטוקאווה",
    spectral: "S",
    shape: "peanut",
    diamM: 535,
    distLd: 5.8,
    distSunAu: 1.324,
    vRel: 29.5,
    rotH: 12.1,
    ecc: 0.28,
    inc: 1.6,
    orbitPhase: 3.8,
    discovery: "1998",
    desc: "יעד Hayabusa. צורת בוטנאד.",
  },
  {
    id: "showcase-eros",
    designation: "433",
    nameHe: "433 ארוס",
    spectral: "S",
    shape: "elongated",
    diamM: 16800,
    distLd: 12.4,
    distSunAu: 1.458,
    vRel: 24.4,
    rotH: 5.3,
    ecc: 0.223,
    inc: 10.8,
    orbitPhase: 0.7,
    discovery: "1898",
    desc: "ה-NEO הראשון שהתגלה. משימת NEAR Shoemaker.",
  },
  {
    id: "showcase-psyche",
    designation: "16",
    nameHe: "16 פסיכה",
    spectral: "M",
    shape: "irregular",
    diamM: 226000,
    distLd: 18,
    distSunAu: 2.923,
    vRel: 17.3,
    rotH: 4.2,
    ecc: 0.134,
    inc: 3.1,
    orbitPhase: 4.2,
    discovery: "1852",
    desc: "עשוי ברובו מברזל וניקל. משימת Psyche.",
  },
  {
    id: "showcase-encke",
    designation: "2P",
    nameHe: "2P/אנקה",
    spectral: "comet",
    shape: "spherical",
    diamM: 4800,
    distLd: 9.5,
    distSunAu: 2.215,
    vRel: 30.3,
    rotH: 11.1,
    ecc: 0.848,
    inc: 11.8,
    orbitPhase: 1.8,
    discovery: "1786",
    desc: "השביט עם התקופה הקצרה ביותר (3.3 שנים). מקור מטאורים טאורידים.",
  },
  {
    id: "showcase-halley",
    designation: "1P",
    nameHe: "1P/האלי",
    spectral: "comet",
    shape: "elongated",
    diamM: 11000,
    distLd: 22,
    distSunAu: 17.8,
    vRel: 54.6,
    rotH: 52.8,
    ecc: 0.967,
    inc: 162,
    orbitPhase: 5.1,
    discovery: '240 לפנה"ס',
    desc: "השביט המפורסם ביותר. חוזר כל 76 שנה.",
  },
  {
    id: "showcase-67p",
    designation: "67P",
    nameHe: "67P/צ'וריומוב",
    spectral: "comet",
    shape: "contact_binary",
    diamM: 4100,
    distLd: 14,
    distSunAu: 3.463,
    vRel: 13.1,
    rotH: 12.4,
    ecc: 0.641,
    inc: 7,
    orbitPhase: 2.9,
    discovery: "1969",
    desc: "יעד Rosetta. צורת כפפת גומי.",
  },
  {
    id: "showcase-toutatis",
    designation: "4179",
    nameHe: "4179 טוטאטיס",
    spectral: "S",
    shape: "irregular",
    diamM: 2450,
    distLd: 7.3,
    distSunAu: 2.512,
    vRel: 29,
    rotH: 130,
    ecc: 0.629,
    inc: 0.4,
    orbitPhase: 3.3,
    discovery: "1989",
    desc: "סיבוב tumbling. חלף קרוב לארץ ב-2004 ו-2012.",
  },
  {
    id: "showcase-hartley2",
    designation: "103P",
    nameHe: "103P/הארטלי 2",
    spectral: "comet",
    shape: "peanut",
    diamM: 2300,
    distLd: 11,
    distSunAu: 3.466,
    vRel: 13.5,
    rotH: 18,
    ecc: 0.695,
    inc: 13.6,
    orbitPhase: 4.7,
    discovery: "1986",
    desc: "שביט פעיל. מפריש CO₂.",
  },
  {
    id: "showcase-vesta",
    designation: "4",
    nameHe: "4 וסטה",
    spectral: "V",
    shape: "spherical",
    diamM: 525000,
    distLd: 16,
    distSunAu: 2.362,
    vRel: 19.3,
    rotH: 5.3,
    ecc: 0.089,
    inc: 7.1,
    orbitPhase: 0.5,
    discovery: "1807",
    desc: "האסטרואיד השני בגודלו. משימת Dawn.",
  },
];

export function getShowcaseEntryForEvent(ev: GlobeAlertEvent): ShowcaseEntry | null {
  if (!ev.showcaseNeo) return null;
  return NEO_SHOWCASE.find((s) => s.id === ev.id) ?? null;
}

export function formatRotationPeriod(rotH: number): string {
  if (rotH < 24) return `${rotH.toFixed(1)} שעות`;
  return `${(rotH / 24).toFixed(1)} ימים`;
}

export function showcaseToGlobeEvent(entry: ShowcaseEntry): GlobeAlertEvent {
  const now = Date.now();
  const approachTime = now + entry.orbitPhase * 1_800_000;
  const idHash = entry.id.split("").reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
  const anchorLon = ((entry.orbitPhase * 137.5 + (idHash % 360)) % 360) - 180;
  const anchorLat = Math.sin(entry.orbitPhase * 1.7 + idHash * 0.01) * Math.max(8, entry.inc);

  return {
    id: entry.id,
    type: "neo",
    lat: anchorLat,
    lon: anchorLon,
    location: entry.nameHe,
    time: now,
    source: "nasa-jpl",
    designation: entry.designation,
    distAu: entry.distSunAu,
    distLd: entry.distLd,
    vRel: entry.vRel,
    approachTime,
    hMagnitude: entry.diamM > 10000 ? 12 : entry.diamM > 1000 ? 18 : 22,
    diameterKm: entry.diamM / 1000,
    isPha: entry.isPha,
    showcaseNeo: true,
    showcaseSpectral: entry.spectral,
    showcaseShape: entry.shape,
    showcaseEcc: entry.ecc,
    showcaseInc: entry.inc,
    showcaseOrbitPhase: entry.orbitPhase,
    showcaseRotH: entry.rotH,
    showcaseDiscovery: entry.discovery,
    showcaseDistSunAu: entry.distSunAu,
    severityText: entry.desc,
  };
}

export function getShowcaseGlobeEvents(): GlobeAlertEvent[] {
  return NEO_SHOWCASE.map(showcaseToGlobeEvent);
}

/** Closed orbit path for floating animation — no visible line drawn. */
export function buildShowcaseOrbitTrack(ev: GlobeAlertEvent): NeoOrbitTrack {
  const entry = getShowcaseEntryForEvent(ev);
  const distLd = entry?.distLd ?? ev.distLd ?? 5;
  const ecc = entry?.ecc ?? ev.showcaseEcc ?? 0.25;
  const inc = entry?.inc ?? ev.showcaseInc ?? 8;
  const phase = entry?.orbitPhase ?? ev.showcaseOrbitPhase ?? 0;
  const steps = 72;
  const points: NeoTrackPoint[] = [];
  const now = Date.now();
  const incRad = (inc * Math.PI) / 180;
  const idHash = ev.id.split("").reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
  const lonOffset = (idHash % 360) * (Math.PI / 180);

  for (let i = 0; i <= steps; i++) {
    const t = phase + (i / steps) * Math.PI * 2 + lonOffset;
    const baseLd = distLd;
    const r = baseLd * (1 + ecc * 0.08 * Math.cos(t));
    const lat = Math.sin(t + phase) * Math.sin(incRad) * 22;
    const lon = (((t * 180) / Math.PI + phase * 42 + idHash * 0.7) % 360) - 180;
    points.push({
      t: now + i * 60_000,
      lat,
      lon,
      distAu: r * (384_400 / 149_597_870.7),
      distLd: r,
      deldotKmS: entry?.vRel ?? ev.vRel ?? 20,
    });
  }

  const closest = points.reduce((a, b) => (a.distLd < b.distLd ? a : b));
  return { designation: ev.designation ?? ev.location, points, closest };
}

export function mergeShowcaseWithReal(real: GlobeAlertEvent[]): GlobeAlertEvent[] {
  const seen = new Set(real.map((e) => (e.designation ?? e.id).toLowerCase()));
  const merged = [...real];
  for (const entry of NEO_SHOWCASE) {
    const key = entry.designation.toLowerCase();
    if (seen.has(key)) continue;
    merged.push(showcaseToGlobeEvent(entry));
    seen.add(key);
  }
  return merged;
}
