import { enrichChannel, enrichRadio, inferChannelLanguages, inferRadioLanguages } from "./languageMetadata";
import { isSpanishMediaChannel, isSpanishMediaRadio } from "./spanishMediaFilter";
import type { Channel, RadioStation } from "./types";

const ALLOWED_LANG = new Set(["heb", "he", "eng", "en"]);
const ENGLISH_COUNTRIES = new Set(["us", "gb", "au", "ca", "ie", "nz"]);
const HEBREW_COUNTRIES = new Set(["il"]);

const FOREIGN_COUNTRY_CODES = new Set([
  "es", "mx", "ar", "co", "cl", "pe", "ve", "uy", "py", "bo", "ec", "gt", "hn", "ni", "pa", "cr", "do", "sv", "cu", "pr", "gq",
  "br", "pt", "fr", "de", "at", "ch", "it", "ru", "ua", "pl", "cz", "sk", "hu", "ro", "bg", "rs", "hr", "si", "tr", "sa", "ae",
  "eg", "iq", "ir", "pk", "in", "bd", "lk", "np", "th", "vn", "id", "my", "ph", "cn", "tw", "hk", "jp", "kr", "za", "ng", "ke",
]);

const HEBREW_SCRIPT = /[\u0590-\u05FF]/;
const CYRILLIC = /[\u0400-\u04FF]/;
const ARABIC_SCRIPT = /[\u0600-\u06FF]/;
const CJK = /[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/;
const DEVANAGARI = /[\u0900-\u097F]/;

const NEWS_NAME_HINT =
  /\b(news|noticias|nachrichten|novosti|nouvelles|actualidad|journal|headlines|breaking\s*news|24\s*news|חדשות|חדשות\s*24)\b/i;
const NEWS_BRAND_HINT =
  /\b(cnn|fox\s*news|bbc\s*news|sky\s*news|msnbc|cnbc|al\s*jazeera|france\s*24|dw\s*news|rt\s*news|globo\s*news|band\s*news|reuters|bloomberg|euronews|ndtv|geo\s*news|ary\s*news)\b/i;

const PORTUGUESE_HINT = /\b(portugu[eê]s|portugues|brasil|brazil|globo|record\s*tv|sbt\b|rede\s*tv|band\s*news)\b/i;

const RELIGIOUS_CLUTTER =
  /\b(quran|islam(ic)?|muslim|shia|sunni|prayer|sermon|church|gospel|christian|jesus|bible|hindu|bhojpuri|bollywood|b4u|zeetv|star\s*plus|sony\s*(tv|max)|urdu|tamil|telugu|bengali|punjabi|marathi|malayalam|kannada|gujarati|ntv\s*(pk|bd|in)|geo\s*tv|ary\s*(digital|news|zindagi))\b/i;

const FOREIGN_LATIN_HINT =
  /\b(french|german|deutsch|italian|turkish|türk|russian|ukrainian|polish|czech|slovak|hungarian|romanian|arabic|chinese|japanese|korean|hindi|persian|farsi|vietnamese|thai|indonesian|malay|filipino|tagalog|dutch|nederlands|swedish|norwegian|danish|finnish|greek|bulgarian|serbian|croatian|slovenian|latvian|lithuanian|estonian)\b/i;

/** Pluto TV repackages many feeds — hide the whole brand from browse/search. */
const PLUTO_TV_NAME_HINT = /\bpluto\s*tv\b/i;

function normCountry(code: string | undefined): string {
  return (code ?? "").trim().toLowerCase();
}

function haystack(parts: Array<string | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

function channelHay(c: Channel): string {
  return haystack([c.name, c.groupTitle, ...(c.tags ?? [])]);
}

function radioHay(r: RadioStation): string {
  return haystack([r.name, ...r.tags]);
}

function resolvedChannelLangs(c: Channel): string[] {
  const ch = c.languages?.length ? c : enrichChannel(c);
  return ch.languages?.length ? ch.languages : inferChannelLanguages(ch);
}

function resolvedRadioLangs(r: RadioStation): string[] {
  const st = r.languages?.length ? r : enrichRadio(r);
  return st.languages?.length ? st.languages : inferRadioLanguages(st);
}

function normLang(code: string): string {
  const t = code.trim().toLowerCase();
  if (t === "he") return "heb";
  if (t === "en") return "eng";
  return t;
}

function hasDisallowedLang(langs: string[]): boolean {
  return langs.some((raw) => {
    const code = normLang(raw);
    if (!code || code === "und") return false;
    return !ALLOWED_LANG.has(code);
  });
}

function hasForeignScript(text: string): boolean {
  if (CYRILLIC.test(text) || ARABIC_SCRIPT.test(text) || CJK.test(text) || DEVANAGARI.test(text)) return true;
  return false;
}

function isPortugueseMedia(hay: string, country: string, langs: string[]): boolean {
  if (PORTUGUESE_HINT.test(hay)) return true;
  if (country === "br" || country === "pt") return true;
  return langs.some((l) => {
    const c = normLang(l);
    return c === "por" || c === "pt";
  });
}

export function isPlutoTvChannel(c: Channel): boolean {
  const hay = channelHay(c);
  if (PLUTO_TV_NAME_HINT.test(hay)) return true;
  const stream = (c.stream ?? "").toLowerCase();
  if (stream.includes("pluto.tv") || stream.includes("pluto.tv/")) return true;
  const tvgId = (c.tvgId ?? "").toLowerCase();
  if (tvgId.includes("pluto")) return true;
  const source = (c.source ?? "").toLowerCase();
  if (source.includes("pluto")) return true;
  return false;
}

export function isNewsMediaChannel(c: Channel): boolean {
  if (c.category === "news") return true;
  const hay = channelHay(c);
  return NEWS_NAME_HINT.test(hay) || NEWS_BRAND_HINT.test(hay);
}

export function isNewsMediaRadio(r: RadioStation): boolean {
  const hay = radioHay(r);
  return NEWS_NAME_HINT.test(hay) || NEWS_BRAND_HINT.test(hay) || /\bnews\b/i.test(hay);
}

export function channelHasHebrew(c: Channel): boolean {
  const langs = resolvedChannelLangs(c);
  if (langs.some((l) => normLang(l) === "heb")) return true;
  if (HEBREW_COUNTRIES.has(normCountry(c.country))) return true;
  return HEBREW_SCRIPT.test(channelHay(c));
}

export function channelHasEnglish(c: Channel): boolean {
  const langs = resolvedChannelLangs(c);
  if (langs.some((l) => normLang(l) === "eng")) return true;
  if (ENGLISH_COUNTRIES.has(normCountry(c.country))) return true;
  const hay = channelHay(c);
  if (HEBREW_SCRIPT.test(hay) || hasForeignScript(hay)) return false;
  if (FOREIGN_LATIN_HINT.test(hay) || isPortugueseMedia(hay, normCountry(c.country), langs)) return false;
  if (isSpanishMediaChannel(c)) return false;
  if (/[a-z]/i.test(hay)) return true;
  return false;
}

export function radioHasHebrew(r: RadioStation): boolean {
  const langs = resolvedRadioLangs(r);
  if (langs.some((l) => normLang(l) === "heb")) return true;
  const country = normCountry(r.countrycode || r.country);
  if (HEBREW_COUNTRIES.has(country)) return true;
  return HEBREW_SCRIPT.test(radioHay(r));
}

export function radioHasEnglish(r: RadioStation): boolean {
  const langs = resolvedRadioLangs(r);
  if (langs.some((l) => normLang(l) === "eng")) return true;
  const country = normCountry(r.countrycode || r.country);
  if (ENGLISH_COUNTRIES.has(country)) return true;
  const hay = radioHay(r);
  if (HEBREW_SCRIPT.test(hay) || hasForeignScript(hay)) return false;
  if (FOREIGN_LATIN_HINT.test(hay) || isPortugueseMedia(hay, country, langs)) return false;
  if (isSpanishMediaRadio(r)) return false;
  if (/[a-z]/i.test(hay)) return true;
  return false;
}

export function channelPassesHeEnCatalog(c: Channel): boolean {
  if (isPlutoTvChannel(c)) return false;
  if (c.category === "religious") return false;
  if (isNewsMediaChannel(c)) return false;

  const ch = c.languages?.length ? c : enrichChannel(c);
  const langs = resolvedChannelLangs(ch);
  const hay = channelHay(ch);
  const country = normCountry(ch.country);

  if (hasForeignScript(hay) && !HEBREW_SCRIPT.test(hay)) return false;
  if (hasDisallowedLang(langs)) return false;
  if (isSpanishMediaChannel(ch)) return false;
  if (isPortugueseMedia(hay, country, langs)) return false;
  if (country && FOREIGN_COUNTRY_CODES.has(country)) return false;
  if (RELIGIOUS_CLUTTER.test(hay)) return false;
  if (FOREIGN_LATIN_HINT.test(hay)) return false;

  return channelHasHebrew(ch) || channelHasEnglish(ch);
}

export function radioPassesHeEnCatalog(r: RadioStation): boolean {
  if (isNewsMediaRadio(r)) return false;

  const st = r.languages?.length ? r : enrichRadio(r);
  const langs = resolvedRadioLangs(st);
  const hay = radioHay(st);
  const country = normCountry(st.countrycode || st.country);

  if (hasForeignScript(hay) && !HEBREW_SCRIPT.test(hay)) return false;
  if (hasDisallowedLang(langs)) return false;
  if (isSpanishMediaRadio(st)) return false;
  if (isPortugueseMedia(hay, country, langs)) return false;
  if (country && FOREIGN_COUNTRY_CODES.has(country)) return false;
  if (RELIGIOUS_CLUTTER.test(hay)) return false;
  if (FOREIGN_LATIN_HINT.test(hay)) return false;

  return radioHasHebrew(st) || radioHasEnglish(st);
}
