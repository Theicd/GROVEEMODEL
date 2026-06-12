/** Shared phrase extraction for routed web search providers. */

const KNOWN_PLACES =
  /(?:ניו\s*יורק|תל\s*א(?:ביב|ב)?|ירושלים|חיפה|לונדון|פריז|paris|london|tokyo|טוקיו|berlin|ברlin|moscow|מוסקבה|rome|רומא|sydney|סידני|miami|מיאמי|los\s*angeles|לוס\s*אנgeles|california|קליפורניה|new\s*zealand|ניו\s*זילנד|אוסטרליה|australia|germany|גרמניה|france|צרפת|japan|יפן|israel|ישראל)/gi;

const COUNTRY_ALIASES: Record<string, string> = {
  ישראל: "Israel",
  ארצות: "United States",
  "ארהב": "United States",
  'ארה"ב': "United States",
  אמריקה: "United States",
  גרמניה: "Germany",
  צרפת: "France",
  בריטניה: "United Kingdom",
  אנגליה: "United Kingdom",
  ספרד: "Spain",
  איטליה: "Italy",
  יפן: "Japan",
  סין: "China",
  הודו: "India",
  ברזיל: "Brazil",
  קנדה: "Canada",
  מקסיקו: "Mexico",
  רוסיה: "Russia",
  אוסטרליה: "Australia",
  "ניו זילנד": "New Zealand",
  מצרים: "Egypt",
  טורקיה: "Turkey",
  פולין: "Poland",
  שבדיה: "Sweden",
  נורווגיה: "Norway",
  פינלנד: "Finland",
  בלגיה: "Belgium",
  הולנד: "Netherlands",
};

export const normalizeCountrySearchName = (raw: string): string => {
  const t = raw.trim().replace(/[?!.]+$/, "");
  const lower = t.toLowerCase();
  for (const [he, en] of Object.entries(COUNTRY_ALIASES)) {
    if (t === he || lower === he.toLowerCase() || lower === en.toLowerCase()) return en;
  }
  return t;
};

/** Place/city/country name from time, weather, marine queries. */
export const extractLocationPhrase = (query: string): string | null => {
  const q = query.trim();
  const patterns: RegExp[] = [
    /(?:מה\s+)?(?:ה)?שע(?:ה|ת)\s+(?:ב|ב־|in|at|for)\s+(.+?)(?:[?!.]?$)/i,
    /(?:what\s+)?time\s+(?:is\s+it\s+)?(?:in|at|for)\s+(.+?)(?:[?!.]?$)/i,
    /(?:מזג\s*האוויר|מז"?\s*א|weather|temperature)\s+(?:ב|ב־|in|at|for)\s+(.+?)(?:[?!.]?$)/i,
    /(?:מה\s+)?(?:מזג\s*האוויר|מז"?\s*א)\s+(?:ב|ב־|של)?\s*(.+?)(?:[?!.]?$)/i,
    /(?:גשם|שלג|מעונן|גשום|rain|snow)\s+(?:ב|ב־|in|at|for)?\s*(.+?)(?:[?!.]?$)/i,
    /(?:צפוי|forecast|expect)\s+(?:גשם|rain|שלג|snow)\s+(?:ב|ב־|in|at|for)?\s*(.+?)(?:[?!.]?$)/i,
    /(?:wave\s*height|גובה\s*גלים|גלים)\s+(?:in|at|near|ב|ב־|ליד)?\s*(.+?)(?:[?!.]?$)/i,
    /(?:ב|in|at)\s+([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,40})(?:[?!.]?$)/i,
  ];
  for (const re of patterns) {
    const m = q.match(re);
    let loc = m?.[1]?.trim().replace(/[?!.]+$/, "").trim();
    if (loc) {
      loc = loc
        .replace(/\s+(temperature|wind|humidity|forecast|today|now|רוח|טמפרatur.*|timezone|time)$/i, "")
        .trim();
    }
    if (loc && loc.length >= 2 && !/^(היום|עכשיו|today|now)$/i.test(loc)) {
      return loc;
    }
  }
  const cityMatch = q.match(KNOWN_PLACES);
  if (cityMatch) return cityMatch[0];
  return null;
};

/** Two places for timezone offset comparison. */
export const extractTimeZonePair = (query: string): [string, string] | null => {
  const q = query.trim();
  const patterns = [
    /(?:כמה\s+)?(?:שעות?\s+)?(?:ה)?(?:פרש|הפרש)\s+(?:יש\s+)?(?:בין|between)\s+(.+?)\s+(?:ל|ו|to|and)\s*(.+?)(?:[?!.]?$)/i,
    /(?:פרש|הפרש)\s+(?:ה)?שע(?:ות)?\s+(?:בין|between)\s+(.+?)\s+(?:ל|ו|to|and)\s*(.+?)(?:[?!.]?$)/i,
    /time\s+(?:zone\s+)?(?:difference|offset)\s+(?:between|from)\s+(.+?)\s+(?:and|to)\s*(.+?)(?:[?!.]?$)/i,
  ];
  for (const re of patterns) {
    const m = q.match(re);
    if (m?.[1] && m[2]) {
      return [m[1].trim(), m[2].trim()];
    }
  }
  return null;
};

/** Country name from country / holiday / government queries. */
export const extractCountryPhrase = (query: string): string | null => {
  const q = query.trim();
  const patterns: RegExp[] = [
    /(?:ב|ב־|in|of|for)\s+(?:מדינת\s+)?([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:מדינ(?:ה|ת)|country)\s+(?:של|of)?\s*([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:בירה|capital|population|currency|מטבע|אוכלוסיה|דגל|flag)\s+(?:של|of)\s+([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:ראש\s+(?:הממשלה|ממשלה|מדינה|המדינה)|נשיא|prime\s+minister|president)\s+(?:של|of|in)\s+([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:מי\s+)?(?:נשיא|ראש\s+(?:הממשלה|ממשלה))\s+([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:חג|חגים|holiday|holidays)\s+(?:ב|ב־|in)\s+([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:האם\s+)?(?:ה)?יום\s+חג\s+(?:ב|ב־|in)\s*([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
    /(?:ב|ב־)([א-ת][א-ת\s\-'".]{2,34})(?:[?!.]?$)/,
    /(?:מידע|info|facts)\s+(?:על|about|on)\s+(?:מדינת\s+)?([A-Za-z\u0590-\u05FF][A-Za-z\u0590-\u05FF\s\-'".]{1,35})(?:[?!.]?$)/i,
  ];
  for (const re of patterns) {
    const m = q.match(re);
    const raw = m?.[1]?.trim().replace(/[?!.]+$/, "").trim();
    if (raw && raw.length >= 2) return normalizeCountrySearchName(raw);
  }
  const known = q.match(
    /\b(Israel|Germany|France|Japan|China|Brazil|Canada|Mexico|Russia|Australia|United States|United Kingdom|Spain|Italy|India|ישראל|גרמניה|צרפת|יפן|סין|ברזיל|קנדה|מקסיקו|רוסיה|אוסטרליה|בריטניה|איטליה|ספרד|הודו|ארצות הברית|ארה"ב)\b/i,
  );
  if (known) return normalizeCountrySearchName(known[0]);
  return null;
};

/** Currency pair e.g. USD to ILS — supports codes, names, and natural Hebrew/English. */
const CURRENCY_CODE = /\b([A-Z]{3})\b/g;

const CURRENCY_ALIASES: Record<string, string> = {
  USD: "USD",
  DOLLAR: "USD",
  DOLLARS: "USD",
  דולר: "USD",
  "דולר אמריקאי": "USD",
  ILS: "ILS",
  NIS: "ILS",
  SHEKEL: "ILS",
  SHEKELS: "ILS",
  שקל: "ILS",
  שקלים: "ILS",
  EUR: "EUR",
  EURO: "EUR",
  EUROS: "EUR",
  יורו: "EUR",
  BRL: "BRL",
  REAL: "BRL",
  REAIS: "BRL",
  ברזילאי: "BRL",
  הברזילאי: "BRL",
  GBP: "GBP",
  POUND: "GBP",
  POUNDS: "GBP",
  לירה: "GBP",
  "לירה שטרלינג": "GBP",
  JPY: "JPY",
  YEN: "JPY",
  ין: "JPY",
  CNY: "CNY",
  YUAN: "CNY",
  RUB: "RUB",
  RUBLE: "RUB",
  CHF: "CHF",
  CAD: "CAD",
  AUD: "AUD",
};

const resolveCurrencyToken = (raw: string): string | null => {
  const t = raw.trim();
  if (!t) return null;
  const upper = t.toUpperCase();
  if (/^[A-Z]{3}$/.test(upper)) return upper;
  const lower = t.toLowerCase();
  for (const [alias, code] of Object.entries(CURRENCY_ALIASES)) {
    if (lower === alias.toLowerCase() || t === alias) return code;
  }
  return null;
};

const collectCurrencyCodes = (query: string): string[] => {
  const found = new Set<string>();
  const upper = query.toUpperCase();
  for (const m of upper.matchAll(CURRENCY_CODE)) {
    found.add(m[1]);
  }
  for (const [alias, code] of Object.entries(CURRENCY_ALIASES)) {
    const re = new RegExp(`(?:^|[\\s,;])${alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:[\\s,;]|$)`, "i");
    if (re.test(query)) found.add(code);
  }
  return [...found];
};

export const extractCurrencyPair = (query: string): { from: string; to: string } | null => {
  const q = query;
  const upper = q.toUpperCase();

  const m1 = upper.match(/\b([A-Z]{3})\b\s*(?:TO|→|->|ל|ב)\s*\b([A-Z]{3})\b/);
  if (m1) return { from: m1[1], to: m1[2] };

  const m2 = upper.match(/(?:שער|RATE|EXCHANGE)\s+(?:OF\s+)?([A-Z]{3})\s+(?:TO|ל)\s+([A-Z]{3})/i);
  if (m2) return { from: m2[1].toUpperCase(), to: m2[2].toUpperCase() };

  if (/(?:דולר|DOLLAR|USD).*(?:שקל|SHEKEL|ILS|NIS)|(?:שקל|SHEKEL|ILS).*(?:דולר|DOLLAR|USD)/i.test(q)) {
    return { from: "USD", to: "ILS" };
  }

  // "כמה BRL אני קונה ב-1 דולר" / "1 dollar to BRL"
  const buyMatch = q.match(
    /(?:כמה|how\s+many)\s+([A-Za-z\u0590-\u05FF]{2,12})\s+(?:אני\s+)?(?:קונה|buy|get).*(?:ב|ב־|for|with)\s*(?:1|אחד|one)\s*([A-Za-z\u0590-\u05FF]{2,12})/i,
  );
  if (buyMatch) {
    const to = resolveCurrencyToken(buyMatch[1]);
    const from = resolveCurrencyToken(buyMatch[2]);
    if (from && to) return { from, to };
  }

  const reverseBuy = q.match(
    /(?:1|אחד|one)\s+([A-Za-z\u0590-\u05FF]{2,12}).*(?:כמה|how\s+many)\s+([A-Za-z\u0590-\u05FF]{2,12})/i,
  );
  if (reverseBuy) {
    const from = resolveCurrencyToken(reverseBuy[1]);
    const to = resolveCurrencyToken(reverseBuy[2]);
    if (from && to) return { from, to };
  }

  if (/(?:יחס|המר|convert|exchange|rate)/i.test(q)) {
    const codes = collectCurrencyCodes(q);
    if (codes.length >= 2) return { from: codes[0], to: codes[1] };
    if (codes.length === 1) {
      const other =
        codes[0] === "USD" ? "ILS" : codes[0] === "BRL" ? "USD" : codes[0] === "EUR" ? "USD" : "USD";
      if (/דולר|dollar|usd/i.test(q) && codes[0] !== "USD") return { from: "USD", to: codes[0] };
      if (/שקל|shekel|ils/i.test(q) && codes[0] !== "ILS") return { from: codes[0], to: "ILS" };
      return { from: other, to: codes[0] };
    }
  }

  const codes = collectCurrencyCodes(q);
  if (codes.length >= 2) return { from: codes[0], to: codes[1] };

  const m3 = q.match(/(?:דולר|dollar|usd|יורו|euro|eur)\s+(?:ל|to|ב)\s+(?:שקל|shekel|ils|nis)/i);
  if (m3 || /דולר.*שקל|usd.*ils/i.test(q)) return { from: "USD", to: "ILS" };

  return null;
};

/** Two places for distance queries: "בין X ל-Y". */
export const extractPlacePair = (query: string): [string, string] | null => {
  const q = query.trim();
  const patterns = [
    /(?:מרחק|distance|כמה\s+(?:ק["″']?מ|km|kilometers?))\s+(?:בין|between)\s+(.+?)\s+(?:ל|ו|to|and)\-?\s*(.+?)(?:[?!.]?$)/i,
    /(?:בין|between)\s+(.+?)\s+(?:ל|ו|to|and)\-?\s*(.+?)(?:[?!.]?$)/i,
  ];
  for (const re of patterns) {
    const m = q.match(re);
    if (m?.[1] && m[2]) {
      const a = m[1].trim().replace(/[?!.]+$/, "");
      const b = m[2].trim().replace(/[?!.]+$/, "");
      if (a.length >= 2 && b.length >= 2) return [a, b];
    }
  }
  return null;
};

/** POI + anchor from "מצא X ליד Y" / "train stations near Heathrow". */
export const extractPoiNearQuery = (query: string): { poi: string; near: string } | null => {
  const q = query.trim();
  const patterns: Array<{ re: RegExp; poi: number; near: number }> = [
    { re: /(?:מצא|find|search\s+for|where\s+is)\s+(.+?)\s+(?:ליד|קרוב\s+ל|near|around|by)\s+(.+?)(?:[?!.]?$)/i, poi: 1, near: 2 },
    { re: /(.+?)\s+(?:ליד|קרוב\s+ל|near|around|by)\s+(.+?)(?:[?!.]?$)/i, poi: 1, near: 2 },
    { re: /(?:אילו|what|which)\s+(.+?)\s+(?:יש|are\s+there|near)\s+(?:ליד|near|at|by)?\s*(.+?)(?:[?!.]?$)/i, poi: 1, near: 2 },
  ];
  for (const { re, poi, near } of patterns) {
    const m = q.match(re);
    const poiText = m?.[poi]?.trim().replace(/[?!.]+$/, "");
    const nearText = m?.[near]?.trim().replace(/[?!.]+$/, "");
    if (poiText && nearText && poiText.length >= 2 && nearText.length >= 2) {
      return { poi: poiText, near: nearText };
    }
  }
  return null;
};

/** News site hint e.g. BBC. */
export const extractNewsSite = (query: string): string | null => {
  if (/bbc/i.test(query)) return "bbc";
  if (/cnn/i.test(query)) return "cnn";
  if (/ynet|ynetnews/i.test(query)) return "ynet";
  if (/חדשות|news|headline|כותרת\s+ראש/i.test(query)) return "bbc";
  return null;
};
