import type { Channel, RadioStation } from "./types";

/** ISO 639-3 / common IPTV codes → display labels. */
export const LANGUAGE_LABELS: Record<string, { he: string; en: string }> = {
  heb: { he: "עברית", en: "Hebrew" },
  he: { he: "עברית", en: "Hebrew" },
  eng: { he: "אנגלית", en: "English" },
  en: { he: "אנגלית", en: "English" },
  ara: { he: "ערבית", en: "Arabic" },
  ar: { he: "ערבית", en: "Arabic" },
  rus: { he: "רוסית", en: "Russian" },
  ru: { he: "רוסית", en: "Russian" },
  fra: { he: "צרפתית", en: "French" },
  fr: { he: "צרפתית", en: "French" },
  deu: { he: "גרמנית", en: "German" },
  de: { he: "גרמנית", en: "German" },
  hin: { he: "הינדי", en: "Hindi" },
  hi: { he: "הינדי", en: "Hindi" },
  urd: { he: "אורדו", en: "Urdu" },
  ur: { he: "אורדו", en: "Urdu" },
  tam: { he: "טמיל", en: "Tamil" },
  tel: { he: "טלוגו", en: "Telugu" },
  ben: { he: "בנגלית", en: "Bengali" },
  spa: { he: "ספרדית", en: "Spanish" },
  es: { he: "ספרדית", en: "Spanish" },
  por: { he: "פורטוגזית", en: "Portuguese" },
  pt: { he: "פורטוגזית", en: "Portuguese" },
  ita: { he: "איטלקית", en: "Italian" },
  it: { he: "איטלקית", en: "Italian" },
  tur: { he: "טורקית", en: "Turkish" },
  tr: { he: "טורקית", en: "Turkish" },
  zho: { he: "סינית", en: "Chinese" },
  zh: { he: "סינית", en: "Chinese" },
  jpn: { he: "יפנית", en: "Japanese" },
  ja: { he: "יפנית", en: "Japanese" },
  kor: { he: "קוריאנית", en: "Korean" },
  ko: { he: "קוריאנית", en: "Korean" },
  pol: { he: "פולנית", en: "Polish" },
  ukr: { he: "אוקראינית", en: "Ukrainian" },
  und: { he: "לא ידוע", en: "Unknown" },
};

const NAME_LANG_HINTS: { code: string; re: RegExp }[] = [
  { code: "hin", re: /\b(hindi|bhojpuri|bollywood|b4u\s*bhojpuri|zee\s*tv|sony\s*tv|star\s*plus)\b/i },
  { code: "urd", re: /\b(urdu|pakistan\s*tv|geo\s*tv|ary\s*)\b/i },
  { code: "ara", re: /\b(arabic|al\s*arabiya|mbc\s|bein\s|quran|islam|muslim|ال[\u0600-\u06FF])/i },
  { code: "heb", re: /\b(hebrew|עברית|כאן|ערוץ\s*1|ערוץ\s*12|reshet|keshet|mako)\b/i },
  { code: "eng", re: /\b(english|bbc|cnn|sky\s|itv|nbc|abc\s|fox\s|hbo|discovery)\b/i },
  { code: "rus", re: /\b(russian|рос|rt\s|russia\s*1|ntv\s)\b/i },
  { code: "fra", re: /\b(french|france\s*24|tf1|canal\+|arte)\b/i },
  { code: "deu", re: /\b(german|deutsch|zdf|ard\s|rtl\s*de)\b/i },
  { code: "spa", re: /\b(spanish|español|espanol|castellano|telemundo|univision|telefe|caracol|televisa|rtve|antena\s*3|movistar|azteca|tudn)\b/i },
  { code: "tur", re: /\b(turkish|türk|trt\s|show\s*turk)\b/i },
];

const ARABIC_SCRIPT = /[\u0600-\u06FF]/;
const HEBREW_SCRIPT = /[\u0590-\u05FF]/;
const DEVANAGARI = /[\u0900-\u097F]/;

function normLangToken(raw: string): string {
  return raw.trim().toLowerCase().replace(/_/g, "-");
}

function addLangCode(out: Set<string>, raw: string): void {
  const t = normLangToken(raw);
  if (!t || t === "und" || t === "undefined") return;
  if (t.length === 2) {
    const map: Record<string, string> = {
      he: "heb",
      en: "eng",
      ar: "ara",
      ru: "rus",
      fr: "fra",
      de: "deu",
      hi: "hin",
      ur: "urd",
      es: "spa",
      pt: "por",
      it: "ita",
      tr: "tur",
      zh: "zho",
      ja: "jpn",
      ko: "kor",
    };
    out.add(map[t] ?? t);
    return;
  }
  if (t.length === 3) out.add(t);
}

function inferFromText(text: string, out: Set<string>): void {
  const lower = text.toLowerCase();
  if (HEBREW_SCRIPT.test(text)) out.add("heb");
  if (ARABIC_SCRIPT.test(text)) out.add("ara");
  if (DEVANAGARI.test(text)) out.add("hin");
  for (const hint of NAME_LANG_HINTS) {
    if (hint.re.test(lower)) out.add(hint.code);
  }
}

function inferFromSource(source: string, out: Set<string>): void {
  const url = source.toLowerCase();
  const lang = url.match(/\/languages\/([a-z]{2,3})\.m3u/)?.[1];
  if (lang) addLangCode(out, lang);
  const country = url.match(/\/countries\/([a-z]{2})\.m3u/)?.[1];
  if (country === "il") out.add("heb");
}

export function inferChannelLanguages(c: Pick<Channel, "name" | "language" | "groupTitle" | "tags" | "category" | "source" | "country">): string[] {
  const out = new Set<string>();
  if (c.language) {
    for (const part of c.language.split(/[,;|/\s]+/)) addLangCode(out, part);
  }
  inferFromSource(c.source, out);
  if (c.country === "il") out.add("heb");
  if (c.country === "us" || c.country === "gb") out.add("eng");
  inferFromText(c.name, out);
  if (c.groupTitle) inferFromText(c.groupTitle, out);
  for (const tag of c.tags ?? []) inferFromText(tag, out);
  if (c.category === "religious" && !out.size) out.add("ara");
  return [...out];
}

export function enrichChannel(c: Channel): Channel {
  const languages = inferChannelLanguages(c);
  const primary = languages[0] || c.language || "";
  const languageSource = c.language ? ("tvg" as const) : languages.length ? ("inferred" as const) : undefined;
  return {
    ...c,
    language: primary || c.language,
    languages: languages.length ? languages : primary ? [primary] : [],
    languageSource,
  };
}

export function inferRadioLanguages(r: Pick<RadioStation, "name" | "language" | "tags" | "countrycode">): string[] {
  const out = new Set<string>();
  if (r.language) {
    for (const part of r.language.split(/[,;|/\s]+/)) addLangCode(out, part);
  }
  if (r.countrycode?.toLowerCase() === "il") out.add("heb");
  inferFromText(r.name, out);
  for (const tag of r.tags ?? []) inferFromText(tag, out);
  return [...out];
}

export function enrichRadio(r: RadioStation): RadioStation {
  const languages = inferRadioLanguages(r);
  const primary = languages[0] || r.language || "";
  return {
    ...r,
    language: primary || r.language,
    languages: languages.length ? languages : primary ? [primary] : [],
  };
}

export function languageDisplayLabel(code: string, he = true): string {
  const labels = LANGUAGE_LABELS[code.toLowerCase()];
  if (labels) return he ? labels.he : labels.en;
  return code.toUpperCase();
}

export function collectLanguageCounts(channels: Channel[], radio: RadioStation[] = []): Map<string, number> {
  const map = new Map<string, number>();
  const bump = (codes: string[] | undefined, fallback: string) => {
    const list = codes?.length ? codes : fallback ? [fallback] : [];
    for (const code of list) {
      if (!code) continue;
      map.set(code, (map.get(code) ?? 0) + 1);
    }
  };
  for (const c of channels) bump(c.languages, c.language);
  for (const r of radio) bump(r.languages, r.language);
  return map;
}
