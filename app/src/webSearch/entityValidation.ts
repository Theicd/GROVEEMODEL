import type { SearchIntent } from "./types";

type ImpossiblePlace = { re: RegExp; labelHe: string };

const IMPOSSIBLE_PLACES: ImpossiblePlace[] = [
  { re: /(?:ה)?ירח|(?:the\s+)?moon\b|lunar\s+surface/i, labelHe: "הירח" },
  { re: /(?:ה)?מאדים|\bmars\b/i, labelHe: "מאדים" },
  { re: /(?:ה)?שמש|\bsun\b(?!\s*day)/i, labelHe: "השמש" },
  { re: /(?:ה)?נוגה|\bvenus\b/i, labelHe: "נוגה" },
  { re: /(?:ה)?צדק|\bjupiter\b/i, labelHe: "צדק" },
  { re: /(?:zodiac|מזל\s+ות)/i, labelHe: "חלל עמוק" },
];

const LIVE_LOCATION_INTENTS: SearchIntent[] = [
  "weather",
  "marine",
  "aviation",
  "ships",
  "places",
  "alerts",
];

export type EntityValidationResult =
  | { ok: true }
  | { ok: false; reasonHe: string; cannedReply: string; summaryHe: string; contextText: string };

const buildNoDataReply = (place: string, domain: string): EntityValidationResult => ({
  ok: false,
  reasonHe: `אין מקור חי ל-${domain} ב-${place}`,
  cannedReply: [
    `אין נתונים חיים ב-${place} — ${domain} (מזג אוויר, ADS-B, AIS) זמינים רק לכדור הארץ.`,
    "נסה שאלה על אזור גיאוגרפי מוגדר (למשל ישראל, לונדון, ים תיכון).",
    `Sources: (none — ${place})`,
  ].join("\n"),
  summaryHe: `חיפוש: אין נתונים ל-${place}`,
  contextText: `[WEB SEARCH — NO LIVE DATA]
The user asked about live ${domain} on ${place}. No ADS-B / weather / AIS source exists there.
RULES:
1. Say clearly in Hebrew that live ${domain} data is not available for ${place} (1–2 sentences).
2. Do NOT invent aircraft counts, weather, or ship numbers.
3. Suggest asking about a region on Earth (Israel, Mediterranean, etc.).
[/WEB SEARCH — NO LIVE DATA]`,
});

export const detectImpossiblePlace = (text: string): string | null => {
  const q = text.trim();
  if (!q) return null;
  for (const { re, labelHe } of IMPOSSIBLE_PLACES) {
    if (re.test(q)) return labelHe;
  }
  return null;
};

export const isAbsurdAviationLocation = (text: string): boolean =>
  detectImpossiblePlace(text) !== null &&
  /(?:מטוס|aircraft|plane|adsb|תעופה|טיס|awacs)/i.test(text);

export const validateLiveDataQuery = (
  query: string,
  intents: SearchIntent[],
): EntityValidationResult => {
  const place = detectImpossiblePlace(query);
  if (!place) return { ok: true };

  const needsLive =
    intents.some((i) => LIVE_LOCATION_INTENTS.includes(i)) ||
    /(?:מטוס|aircraft|weather|מזג|ספינ|ship|ais)/i.test(query);

  if (!needsLive) return { ok: true };

  if (intents.includes("aviation") || isAbsurdAviationLocation(query)) {
    return buildNoDataReply(place, "תעופה (ADS-B)");
  }
  if (intents.includes("weather") || intents.includes("marine")) {
    return buildNoDataReply(place, "מזג אוויר / ים");
  }
  if (intents.includes("ships")) {
    return buildNoDataReply(place, "כלי שייט (AIS)");
  }
  if (intents.includes("places")) {
    return buildNoDataReply(place, "מיקומים");
  }

  return buildNoDataReply(place, "נתונים חיים");
};
