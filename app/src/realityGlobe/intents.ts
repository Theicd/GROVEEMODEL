import type { GlobeCommand } from "./bridge";
import { normalizeCountrySearchName } from "../webSearch/queryExtract";

const PLACE_PATTERNS = [
  /(?:הצג(?:ה|י)?|הראה|תציג|הציג|show)\s+(?:לי\s+|me\s+|א(?:ת|ת ה)?\s*)?(.+?)(?:\s+על\s+המפה|\s+on\s+the\s+map|\?|$)/i,
  /(?:ע(?:ל|ל\s+)?(?:ה)?מפה\s+(?:א(?:ת|ת ה)?\s*)?)(.+?)(?:\?|$)/i,
  /(?:איפה|היכן|where\s+is)\s+(?:נמצא(?:ים|ת)?\s+)?(.+?)(?:\?|$)/i,
  /(?:מיקום|location\s+of)\s+(.+?)(?:\?|$)/i,
  /(?:האם\s+)?(?:אתה\s+)?(?:יכול\s+)?(?:להציג|להראות|לshow)\s+(?:לי\s+)?(?:ע(?:ל|ל\s+)?(?:ה)?מפה\s+)?(?:א(?:ת|ת ה)?\s*)?(.+?)(?:\?|$)/i,
];

const KNOWN_PLACES = [
  "חיפה",
  "ירושלים",
  "תל אביב",
  "באר שבע",
  "אילת",
  "haifa",
  "jerusalem",
  "tel aviv",
  "london",
  "paris",
  "berlin",
  "new york",
  "גרמניה",
  "germany",
  "צרפת",
  "france",
  "מוסקבה",
  "moscow",
  "רומא",
  "rome",
  "טוקיו",
  "tokyo",
];

const COUNTRY_IN_QUERY =
  /(?:גרמניה|צרפת|ישראל|germany|france|israel|spain|ספרד|italy|איטליה|japan|יפן|china|סין|brazil|ברזיל|canada|קנדה|mexico|מקסיקו|russia|רוסיה|australia|אוסטרליה|egypt|מצרים|turkey|טורקיה|greece|יוון|jordan|ירדן|lebanon|לבנון|poland|פולין|netherlands|הולנד|belgium|בלגיה|sweden|שבדיה|norway|נורווגיה|finland|פינלנד|united\s+states|ארצות\s+הברית|ארה"ב|united\s+kingdom|בריטניה|אנגליה)/i;

function cleanExtractedPlace(raw: string): string {
  return raw
    .replace(/^(\s*א(?:ת|ת ה)?\s*)/i, "")
    .replace(/^(\s*the\s+)/i, "")
    .replace(/\s+על\s+המפה.*/i, "")
    .replace(/\s+on\s+the\s+map.*/i, "")
    .trim();
}

function extractCountryToken(query: string): string | null {
  const m = query.match(COUNTRY_IN_QUERY);
  if (!m?.[0]) return null;
  return normalizeCountrySearchName(m[0]);
}

export function isGlobePresentationQuery(query: string): boolean {
  return /על\s+המפה|on\s+the\s+map|הצג|הציג|תציג|הראה\s+(?:לי\s+)?|show\s+(?:me\s+)?/i.test(query);
}

function extractPlaceName(query: string): string | null {
  for (const re of PLACE_PATTERNS) {
    const m = query.match(re);
    if (m?.[1]) {
      const p = cleanExtractedPlace(m[1]);
      if (p.length >= 2) return normalizeCountrySearchName(p);
    }
  }
  const country = extractCountryToken(query);
  if (country) return country;
  const lower = query.toLowerCase();
  for (const p of KNOWN_PLACES) {
    if (lower.includes(p.toLowerCase())) return normalizeCountrySearchName(p);
  }
  return null;
}

const GLOBE_INTENTS = new Set([
  "earthquake",
  "aviation",
  "satellite",
  "places",
  "distance",
  "weather",
  "marine",
  "spaceweather",
  "israel-alerts",
  "disasters",
]);

export function shouldOpenGlobePanel(query: string, intents: string[] = []): boolean {
  const q = query.trim();
  if (!q) return false;
  if (intents.some((i) => GLOBE_INTENTS.has(i))) return true;
  return /גלובוס|עולם\s*חי|על\s+המפה|reality|real.?time\s*map|show.*map|איפה|היכן|where\s+is|הראה\s+לי|הצג|הציג|תציג|רעיד|earthquake|מטוס|aircraft|לוויין|satellite|iss|התרע|צבע\s*אדום|סופה|hurricane|typhoon/i.test(
    q,
  );
}

export function buildGlobeCommand(query: string, intents: string[] = []): GlobeCommand | null {
  const q = query.trim();
  if (!q) return null;

  if (/רעיד|earthquake|רעש\s*אדמה|seismic/i.test(q) || intents.includes("earthquake")) {
    return { type: "focusEarthquakes" };
  }
  if (/מטוס|aircraft|adsb|תעופה|plane/i.test(q) || intents.includes("aviation")) {
    return { type: "showLayer", layer: "aviation" };
  }
  if (/\biss\b|לוויין|satellite|חלל/i.test(q) || intents.includes("satellite")) {
    return { type: "showLayer", layer: "iss" };
  }
  if (/התרע|צבע\s*אדום|tzeva|oref/i.test(q) || intents.includes("israel-alerts")) {
    return { type: "focusIsrael" };
  }
  if (/סופה|הurricane|typhoon|מזג\s*אוויר|weather/i.test(q) || intents.includes("weather")) {
    return { type: "showLayer", layer: "weather" };
  }
  if (/ספינ|ship|marine|ים/i.test(q) || intents.includes("marine")) {
    return { type: "showLayer", layer: "marine" };
  }

  const place = extractPlaceName(q);
  if (place) {
    const presentation = isGlobePresentationQuery(q) || /על\s+המפה|on\s+the\s+map/i.test(q);
    return { type: "focusPlaceQuiet", name: place, presentation };
  }

  if (intents.includes("places") || intents.includes("distance")) {
    return { type: "globe3d" };
  }

  if (shouldOpenGlobePanel(q, intents)) return { type: "globe3d" };
  return null;
}
