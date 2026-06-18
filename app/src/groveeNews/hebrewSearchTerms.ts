import { normalizeNewsEngineQuery } from "./newsQueryNormalize";

const HEBREW_CHAR = /[\u0590-\u05FF]/;

const HEBREW_QUERY_STOP = new Set([
  "חדשות",
  "כותרות",
  "כתבות",
  "דיווחים",
  "מה",
  "מי",
  "איך",
  "מתי",
  "איפה",
  "למה",
  "של",
  "על",
  "את",
  "זה",
  "היא",
  "הוא",
  "עם",
  "גם",
  "כרגע",
  "היום",
  "עכשיו",
  "בעולם",
  "בישראל",
  "חפש",
  "מצא",
  "תביא",
]);

/** English search term → Hebrew headline variants for cross-language RSS matching. */
export const ENGLISH_TO_HEBREW_ALIASES: Record<string, string[]> = {
  israel: ["ישראל", "ישראלי", "ישראלית", "צה״ל", "צהל"],
  israeli: ["ישראל", "ישראלי"],
  iran: ["איראן", "אירני"],
  gaza: ["עזה"],
  hamas: ["חמאס"],
  netanyahu: ["נתניהו", "ביבי"],
  ukraine: ["אוקראינה", "אוקראיני"],
  russia: ["רוסיה", "רוסי"],
  china: ["סין", "סיני"],
  trump: ["טראמפ", "דונלד"],
  biden: ["ביידן"],
  jerusalem: ["ירושלים"],
  "tel aviv": ["תל אביב", "תל-אביב"],
  war: ["מלחמה", "מלחמת"],
  military: ["צבא", "צה״ל", "צהל", "ביטחון"],
  defense: ["ביטחון", "צה״ל", "צהל"],
  economy: ["כלכלה", "כלכלי"],
  market: ["שוק", "בורסה"],
  politics: ["פוליטיקה", "פוליטי", "ממשלה", "כנסת"],
  election: ["בחירות"],
  protest: ["מחאה", "מחאות"],
  space: ["חלל", "חללית"],
  nasa: ["נאס״א", "נאסא"],
  weather: ["מזג", "אוויר"],
  health: ["בריאות", "רפואה"],
  sport: ["ספורט"],
  football: ["כדורגל"],
  technology: ["טכנולוגיה", "הייטק"],
  startup: ["סטארטאפ", "סטארטאפים"],
  cyber: ["סייבר"],
  bitcoin: ["ביטקוין", "קריפטו"],
  crypto: ["קריפטו", "ביטקוין"],
};

export function extractHebrewQueryTokens(query: string): string[] {
  if (!HEBREW_CHAR.test(query)) return [];
  return [
    ...new Set(
      query
        .replace(/[^\u0590-\u05FF\s]/g, " ")
        .split(/\s+/)
        .map((t) => t.trim())
        .filter((t) => t.length >= 2 && !HEBREW_QUERY_STOP.has(t)),
    ),
  ];
}

/** Build Hebrew + English terms for matching Hebrew RSS headlines from Hebrew or mixed queries. */
export function buildBilingualSearchTerms(query: string, aiKeywords: string[] = []): string[] {
  const out = new Set<string>();
  const raw = query.trim();
  if (!raw) return [];

  for (const t of extractHebrewQueryTokens(raw)) out.add(t);

  const engine = normalizeNewsEngineQuery(raw);
  const englishTerms = engine
    .toLowerCase()
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 1);

  for (const t of englishTerms) {
    out.add(t);
    for (const he of ENGLISH_TO_HEBREW_ALIASES[t] ?? []) out.add(he);
    if (t.includes(" ")) {
      for (const part of t.split(/\s+/)) {
        for (const he of ENGLISH_TO_HEBREW_ALIASES[part] ?? []) out.add(he);
      }
    }
  }

  for (const k of aiKeywords) {
    const kl = k.toLowerCase().trim();
    if (kl.length > 1) {
      out.add(kl);
      for (const he of ENGLISH_TO_HEBREW_ALIASES[kl] ?? []) out.add(he);
    }
  }

  return [...out];
}

export function queryHasHebrew(text: string): boolean {
  return HEBREW_CHAR.test(text);
}
