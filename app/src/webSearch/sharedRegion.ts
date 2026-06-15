import { geocodePlace, formatPlaceLabel, type GeoPlace } from "./geoResolve";
import { extractLocationPhrase } from "./queryExtract";
import { isCrossSourceQuery } from "./crossSourceIntents";
import type { SearchIntent, SharedSearchRegion } from "./types";

export type { SharedSearchRegion };

const cache = new Map<string, { at: number; region: SharedSearchRegion | null }>();
const CACHE_TTL_MS = 10 * 60_000;

const REGION_ALIASES: Array<{ re: RegExp; phrase: string }> = [
  { re: /ישראל|israel|tel\s*aviv|תל\s*אביב|נתב"?ג/i, phrase: "ישראל" },
  { re: /לונדון|london/i, phrase: "London" },
  { re: /(?:ה)?ים\s*תיכון|mediterranean|med\s+sea/i, phrase: "Mediterranean Sea" },
  { re: /מפרץ\s+חיפה|haifa\s+bay/i, phrase: "Haifa" },
  { re: /חיפה|haifa/i, phrase: "Haifa" },
  { re: /פריז|paris/i, phrase: "Paris" },
  { re: /ניו\s*יורק|new\s*york/i, phrase: "New York" },
  { re: /יוון|greece/i, phrase: "Greece" },
  { re: /רוטרדם|rotterdam/i, phrase: "Rotterdam" },
];

export const shouldResolveSharedRegion = (query: string, intents: SearchIntent[]): boolean => {
  const geoIntents = intents.filter((i) =>
    ["weather", "aviation", "ships", "marine", "airquality", "alerts"].includes(i),
  );
  return isCrossSourceQuery(query) || geoIntents.length >= 2;
};

export const extractRegionPhrase = (query: string): string | null => {
  for (const { re, phrase } of REGION_ALIASES) {
    if (re.test(query)) return phrase;
  }

  const fromPatterns = extractLocationPhrase(query);
  if (fromPatterns && fromPatterns.length >= 2) return fromPatterns;

  const crossM = query.match(
    /(?:מעל|above|over|באזור|באזור\s+ה)?(?:של\s+)?(ישראל|israel|לונדון|london|ים\s+תיכון|mediterranean|חיפה|haifa|פריז|paris)/i,
  );
  if (crossM?.[1]) {
    const hit = REGION_ALIASES.find(({ re }) => re.test(crossM[1]));
    return hit?.phrase ?? crossM[1].trim();
  }

  return fromPatterns;
};

export const resolveSharedSearchRegion = async (
  query: string,
  intents: SearchIntent[],
): Promise<SharedSearchRegion | null> => {
  if (!shouldResolveSharedRegion(query, intents)) return null;

  const phrase = extractRegionPhrase(query);
  if (!phrase || phrase.length < 2) return null;

  const key = phrase.trim().toLowerCase();
  const cached = cache.get(key);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.region;

  const place = await geocodePlace(phrase);
  const region: SharedSearchRegion | null = place
    ? { label: formatPlaceLabel(place), place, phrase }
    : null;

  cache.set(key, { at: Date.now(), region });
  return region;
};

export const clearSharedRegionCache = (): void => {
  cache.clear();
};

/** Small bbox (~1°) around geocoded center for AIS / regional providers. */
export const bboxAroundPlace = (
  place: GeoPlace,
  delta = 0.45,
): { minLat: number; maxLat: number; minLon: number; maxLon: number } => ({
  minLat: place.latitude - delta,
  maxLat: place.latitude + delta,
  minLon: place.longitude - delta,
  maxLon: place.longitude + delta,
});
