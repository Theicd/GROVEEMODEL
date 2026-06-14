import { geocodePlace } from "../webSearch/geoResolve";
import type { ShipBbox } from "./medPorts";

export type ShipRegion = {
  label: string;
  bbox: ShipBbox | null;
  center: { lat: number; lon: number } | null;
  radiusNm: number;
};

const SUEZ_BBOX: ShipBbox = { minLat: 29.8, maxLat: 31.55, minLon: 32.15, maxLon: 33.05 };
const MED_BBOX: ShipBbox = { minLat: 27, maxLat: 42, minLon: 18, maxLon: 38 };
export const HAIFA_BAY_BBOX: ShipBbox = { minLat: 32.72, maxLat: 32.92, minLon: 34.92, maxLon: 35.12 };
const ASHDOD_BBOX: ShipBbox = { minLat: 31.75, maxLat: 31.95, minLon: 34.55, maxLon: 34.75 };
const EILAT_BBOX: ShipBbox = { minLat: 29.45, maxLat: 29.65, minLon: 34.88, maxLon: 35.05 };

const REGION_PRESETS: Record<string, Omit<ShipRegion, "center"> & { center: { lat: number; lon: number } }> = {
  suez: { label: "תעלת סואץ", bbox: SUEZ_BBOX, center: { lat: 30.5, lon: 32.35 }, radiusNm: 120 },
  haifa: { label: "מפרץ חיפה", bbox: HAIFA_BAY_BBOX, center: { lat: 32.82, lon: 35.0 }, radiusNm: 60 },
  med: { label: "ים תיכון", bbox: MED_BBOX, center: { lat: 34.0, lon: 28.0 }, radiusNm: 500 },
  ashdod: { label: "נמל אשדוד", bbox: ASHDOD_BBOX, center: { lat: 31.84, lon: 34.64 }, radiusNm: 50 },
  eilat: { label: "מפרץ אילat", bbox: EILAT_BBOX, center: { lat: 29.55, lon: 34.95 }, radiusNm: 50 },
  rotterdam: {
    label: "נמל רוטרדם",
    bbox: { minLat: 51.75, maxLat: 52.05, minLon: 3.8, maxLon: 4.75 },
    center: { lat: 51.92, lon: 4.48 },
    radiusNm: 90,
  },
  persian: {
    label: "מפרץ הפרסי",
    bbox: { minLat: 24.0, maxLat: 30.5, minLon: 48.0, maxLon: 57.5 },
    center: { lat: 27.0, lon: 52.0 },
    radiusNm: 350,
  },
  hamburg: {
    label: "נמל המבורג",
    bbox: { minLat: 53.4, maxLat: 53.65, minLon: 9.7, maxLon: 10.2 },
    center: { lat: 53.55, lon: 9.99 },
    radiusNm: 80,
  },
  piraeus: {
    label: "נמל פיראוס",
    bbox: { minLat: 37.85, maxLat: 38.05, minLon: 23.55, maxLon: 23.75 },
    center: { lat: 37.94, lon: 23.64 },
    radiusNm: 70,
  },
  singapore: {
    label: "מצר סингapore",
    bbox: { minLat: 1.1, maxLat: 1.45, minLon: 103.6, maxLon: 104.1 },
    center: { lat: 1.25, lon: 103.85 },
    radiusNm: 80,
  },
};

const COUNTRY_BBOXES: Record<string, { label: string; bbox: ShipBbox; center: { lat: number; lon: number } }> = {
  greece: {
    label: "יוון (חופים)",
    bbox: { minLat: 34.8, maxLat: 41.5, minLon: 19.3, maxLon: 28.5 },
    center: { lat: 38.5, lon: 23.7 },
  },
  netherlands: {
    label: "הולנד (חופים)",
    bbox: { minLat: 51.0, maxLat: 53.6, minLon: 3.0, maxLon: 7.5 },
    center: { lat: 52.2, lon: 5.3 },
  },
  israel: {
    label: "ישראל (חוף)",
    bbox: { minLat: 29.4, maxLat: 33.4, minLon: 34.2, maxLon: 35.9 },
    center: { lat: 31.8, lon: 34.8 },
  },
  turkey: {
    label: "טורקיה (חופים)",
    bbox: { minLat: 35.5, maxLat: 42.2, minLon: 25.5, maxLon: 36.5 },
    center: { lat: 39.0, lon: 31.0 },
  },
};

const PORT_ALIASES: Array<{ re: RegExp; key: keyof typeof REGION_PRESETS }> = [
  { re: /רוטרדם|rotterdam/i, key: "rotterdam" },
  { re: /המבורג|hamburg/i, key: "hamburg" },
  { re: /פיראוס|piraeus/i, key: "piraeus" },
  { re: /סינגapore|singapore/i, key: "singapore" },
  { re: /אשדוד|ashdod/i, key: "ashdod" },
  { re: /אילat|eilat|עקב/i, key: "eilat" },
];

const COUNTRY_ALIASES: Array<{ re: RegExp; key: keyof typeof COUNTRY_BBOXES }> = [
  { re: /יוון|greece|hellas|ελλάδα/i, key: "greece" },
  { re: /הולנד|netherlands|holland/i, key: "netherlands" },
  { re: /ישראל|israel/i, key: "israel" },
  { re: /טורקיה|turkey|türkiye/i, key: "turkey" },
];

export const detectRegionPreset = (query: string): keyof typeof REGION_PRESETS | null => {
  if (/פרסי|persian\s+gulf|מפרץ\s+הפרס/i.test(query)) return "persian";
  if (/סואץ|suez\s*canal|תעלת\s+סואץ/i.test(query)) return "suez";
  if (/מפרץ\s*חיפה|haifa\s+bay/i.test(query) || (/חיפה|haifa/i.test(query) && /(?:מפרץ|ספינ|אונi|ship|שייט|vessel|מצופ|מגדלור)/i.test(query))) {
    return "haifa";
  }
  if (/ים\s+תיכון|mediterranean/i.test(query)) return "med";
  for (const { re, key } of PORT_ALIASES) {
    if (re.test(query)) return key;
  }
  return null;
};

const detectCountryRegion = (query: string): (typeof COUNTRY_BBOXES)[keyof typeof COUNTRY_BBOXES] | null => {
  for (const { re, key } of COUNTRY_ALIASES) {
    if (re.test(query)) return COUNTRY_BBOXES[key];
  }
  return null;
};

const extractPlacePhrase = (query: string): string | null => {
  const patterns = [
    /(?:אוניות|ספינות|כלי\s+שייט|ships?|vessels?)\s+(?:ליד|ב|סביב|near|around|in)\s+([^\?.,!]{2,48})/i,
    /(?:מצופים|מגדלורים|buoys?|lighthouses?)\s+(?:ליד|ב|near|around|in)\s+([^\?.,!]{2,48})/i,
    /(?:ב|ליד|סביב|near|around|in)\s+([^\?.,!]{2,48})/i,
    /(?:מפרץ|נמל|port\s+of|harbour\s+of|harbor\s+of)\s+([^\?.,!]{2,40})/i,
  ];
  for (const re of patterns) {
    const m = query.match(re);
    const phrase = m?.[1]?.trim();
    if (phrase && phrase.length >= 2) {
      return phrase.replace(/\s+(?:עכשיו|כרגע|now)$/i, "").trim();
    }
  }
  return null;
};

const bboxAround = (lat: number, lon: number, delta = 0.35): ShipBbox => ({
  minLat: lat - delta,
  maxLat: lat + delta,
  minLon: lon - delta,
  maxLon: lon + delta,
});

/** Resolve label, bbox, and AIS search center from a natural-language query. */
export const resolveShipRegion = async (query: string): Promise<ShipRegion> => {
  const presetKey = detectRegionPreset(query);
  if (presetKey) {
    const p = REGION_PRESETS[presetKey];
    return { label: p.label, bbox: p.bbox, center: p.center, radiusNm: p.radiusNm };
  }

  const country = detectCountryRegion(query);
  if (country) {
    return {
      label: country.label,
      bbox: country.bbox,
      center: country.center,
      radiusNm: 200,
    };
  }

  const phrase = extractPlacePhrase(query);
  if (phrase) {
    for (const { re, key } of PORT_ALIASES) {
      if (re.test(phrase)) {
        const p = REGION_PRESETS[key];
        return { label: p.label, bbox: p.bbox, center: p.center, radiusNm: p.radiusNm };
      }
    }
    for (const { re, key } of COUNTRY_ALIASES) {
      if (re.test(phrase)) {
        const c = COUNTRY_BBOXES[key];
        return { label: c.label, bbox: c.bbox, center: c.center, radiusNm: 200 };
      }
    }

    const geo = await geocodePlace(phrase);
    if (geo) {
      const delta = geo.country_code && phrase.length <= 12 ? 1.2 : 0.45;
      const bbox = bboxAround(geo.latitude, geo.longitude, delta);
      return {
        label: phrase,
        bbox,
        center: { lat: geo.latitude, lon: geo.longitude },
        radiusNm: delta > 1 ? 250 : 90,
      };
    }
  }

  return { label: "גלובלי (Digitraffic)", bbox: null, center: null, radiusNm: 500 };
};

export { MED_BBOX, SUEZ_BBOX };
