import { enrichChannel, enrichRadio, inferChannelLanguages, inferRadioLanguages } from "./languageMetadata";
import type { Channel, RadioStation } from "./types";

/** ISO 3166-1 — Spanish-primary broadcast regions (not pt-BR). */
const SPANISH_COUNTRY_CODES = new Set([
  "es",
  "mx",
  "ar",
  "co",
  "cl",
  "pe",
  "ve",
  "uy",
  "py",
  "bo",
  "ec",
  "gt",
  "hn",
  "ni",
  "pa",
  "cr",
  "do",
  "sv",
  "cu",
  "pr",
  "gq",
]);

const SPANISH_LANG_CODES = new Set(["spa", "es", "esp", "spanish", "español", "espanol", "castellano"]);

/** Strong Spanish channel / network name signals. */
const SPANISH_NAME_HINT =
  /\b(españa|español|espanol|castellano|telemundo|univision|univisi[oó]n|galavisi[oó]n|antena\s*3|la\s*1\b|rtve|movistar|caracol|telefe|televisa|azteca|las\s*estrellas|tudn|discovery\s*en\s*español|españa|mexico|méxico|argentina|colombia|chile|perú|peru|venezuela|ecuador|guatemala|honduras|nicaragua|panama|panamá|costa\s*rica|republica\s*dominicana|dominicana|uruguay|paraguay|bolivia|noticias\s*24|deportes\s*en\s*vivo|canal\s*\d{1,2}\b|tv\s*españa)\b/i;

/** Portuguese — avoid hiding pt-BR / pt-PT as Spanish. */
const PORTUGUESE_NAME_HINT =
  /\b(portugu[eê]s|portugues|brasil|brazil|globo|record\s*tv|sbt\s|band\s*news|rede\s*tv)\b/i;

const SPANISH_STRONG_CHARS = /[ñ¿¡]/;
const SPANISH_WEAK_CHARS = /[áéíóúü]/;

function normCountry(code: string | undefined): string {
  return (code ?? "").trim().toLowerCase();
}

function languagesIncludeSpanish(langs: string[] | undefined, fallback: string): boolean {
  const list = langs?.length ? langs : fallback ? [fallback.split(/[,;|/\s]+/)[0] ?? ""] : [];
  return list.some((raw) => {
    const code = raw.trim().toLowerCase();
    if (!code) return false;
    return SPANISH_LANG_CODES.has(code) || code.startsWith("spa");
  });
}

function haystack(parts: Array<string | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

function textLooksSpanish(hay: string): boolean {
  if (!hay.trim()) return false;
  if (PORTUGUESE_NAME_HINT.test(hay)) return false;
  if (SPANISH_NAME_HINT.test(hay)) return true;
  if (SPANISH_STRONG_CHARS.test(hay)) return true;
  if (SPANISH_WEAK_CHARS.test(hay) && /\b(la|el|los|las|del|de|en|y|canal|noticias|deportes|pel[ií]culas|música|musica|niñ[oa]s|vivo)\b/i.test(hay)) {
    return true;
  }
  return false;
}

export function isSpanishMediaChannel(c: Channel): boolean {
  const ch = c.languages?.length ? c : enrichChannel(c);

  if (languagesIncludeSpanish(ch.languages, ch.language)) return true;

  const country = normCountry(ch.country);
  if (country && SPANISH_COUNTRY_CODES.has(country)) return true;

  const hay = haystack([ch.name, ch.groupTitle, ...(ch.tags ?? [])]);
  if (textLooksSpanish(hay)) return true;

  const inferred = inferChannelLanguages(ch);
  if (inferred.includes("spa")) return true;

  return false;
}

export function isSpanishMediaRadio(r: RadioStation): boolean {
  const st = r.languages?.length ? r : enrichRadio(r);

  if (languagesIncludeSpanish(st.languages, st.language)) return true;

  const country = normCountry(st.countrycode || st.country);
  if (country && SPANISH_COUNTRY_CODES.has(country)) return true;

  const hay = haystack([st.name, ...st.tags]);
  if (textLooksSpanish(hay)) return true;

  const inferred = inferRadioLanguages(st);
  if (inferred.includes("spa")) return true;

  return false;
}
