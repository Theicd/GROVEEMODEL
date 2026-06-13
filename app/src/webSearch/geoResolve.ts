import { fetchJson } from "./fetchJson";

export type GeoPlace = {
  name: string;
  latitude: number;
  longitude: number;
  elevation?: number;
  country_code?: string;
  admin1?: string;
  timezone?: string;
};

type GeoResult = {
  results?: GeoPlace[];
};

/** Fast fallback when Open-Meteo geocoding is slow or fails. */
const KNOWN_PLACES: Record<string, GeoPlace> = {
  "תל אביב": { name: "Tel Aviv", latitude: 32.0853, longitude: 34.7818, country_code: "IL", admin1: "Tel Aviv" },
  "תל אביב יפו": { name: "Tel Aviv", latitude: 32.0853, longitude: 34.7818, country_code: "IL", admin1: "Tel Aviv" },
  "tel aviv": { name: "Tel Aviv", latitude: 32.0853, longitude: 34.7818, country_code: "IL", admin1: "Tel Aviv" },
  "ירושלים": { name: "Jerusalem", latitude: 31.7683, longitude: 35.2137, country_code: "IL" },
  jerusalem: { name: "Jerusalem", latitude: 31.7683, longitude: 35.2137, country_code: "IL" },
  "חיפה": { name: "Haifa", latitude: 32.794, longitude: 34.9896, country_code: "IL" },
  haifa: { name: "Haifa", latitude: 32.794, longitude: 34.9896, country_code: "IL" },
  "לונדון": { name: "London", latitude: 51.50853, longitude: -0.12574, country_code: "GB", admin1: "England" },
  london: { name: "London", latitude: 51.50853, longitude: -0.12574, country_code: "GB", admin1: "England" },
  "פריז": { name: "Paris", latitude: 48.85341, longitude: 2.3488, country_code: "FR" },
  paris: { name: "Paris", latitude: 48.85341, longitude: 2.3488, country_code: "FR" },
  "ניו יורק": { name: "New York", latitude: 40.71427, longitude: -74.00597, country_code: "US", admin1: "New York" },
  "new york": { name: "New York", latitude: 40.71427, longitude: -74.00597, country_code: "US", admin1: "New York" },
  "טוקיו": { name: "Tokyo", latitude: 35.6895, longitude: 139.6917, country_code: "JP" },
  tokyo: { name: "Tokyo", latitude: 35.6895, longitude: 139.6917, country_code: "JP" },
  madrid: { name: "Madrid", latitude: 40.4165, longitude: -3.70256, country_code: "ES" },
};

const normalizePlaceKey = (name: string): string => name.trim().toLowerCase().replace(/\s+/g, " ");

const lookupKnownPlace = (name: string): GeoPlace | null => {
  const key = normalizePlaceKey(name);
  if (KNOWN_PLACES[key]) return KNOWN_PLACES[key];
  for (const [k, place] of Object.entries(KNOWN_PLACES)) {
    if (key.includes(k) || k.includes(key)) return place;
  }
  return null;
};

const geocodeViaApi = async (name: string, language: "he" | "en"): Promise<GeoPlace | null> => {
  const encoded = encodeURIComponent(name.trim());
  const data = await fetchJson<GeoResult>(
    `https://geocoding-api.open-meteo.com/v1/search?name=${encoded}&count=3&language=${language}&format=json`,
    undefined,
    { timeoutMs: 6_000 },
  );
  return data.results?.[0] ?? null;
};

/** Open-Meteo geocoding — shared by weather, marine, world time. */
export const geocodePlace = async (name: string): Promise<GeoPlace | null> => {
  const q = name.trim();
  if (q.length < 2) return null;

  const known = lookupKnownPlace(q);
  if (known) return known;

  try {
    const he = await geocodeViaApi(q, "he");
    if (he) return he;
  } catch {
    /* retry en */
  }

  try {
    return await geocodeViaApi(q, "en");
  } catch {
    return lookupKnownPlace(q);
  }
};

export const formatPlaceLabel = (place: GeoPlace): string =>
  [place.name, place.admin1, place.country_code].filter(Boolean).join(", ");
