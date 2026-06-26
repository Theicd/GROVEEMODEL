import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { TmdbLanguage } from "./tmdbLocale";

/** Text is Latin-only (no Hebrew) — worth auto-translating for Hebrew UI. */
export function needsHebrewTranslation(text: string | null | undefined): boolean {
  const t = text?.trim();
  if (!t) return false;
  if (/[\u0590-\u05ff]/.test(t)) return false;
  return /[a-z]/i.test(t);
}

/** Fill missing Hebrew overview from Google Translate — keep titles from TMDB/EPG. */
export async function localizeTmdbMetaForUi<T extends { title: string; overview: string | null }>(
  meta: T,
  language: TmdbLanguage,
): Promise<T> {
  if (language !== "he-IL") return meta;

  if (!needsHebrewTranslation(meta.overview)) return meta;

  try {
    const { texts: translated } = await translateTexts([meta.overview!], "he", "en");
    const overview = translated[0]?.trim() || meta.overview;
    return { ...meta, overview };
  } catch {
    return meta;
  }
}

/** Batch-translate movie search hits for Hebrew UI. */
export async function localizeMovieHitsForHebrew<
  T extends { title: string; snippet?: string },
>(hits: T[]): Promise<T[]> {
  if (!hits.length) return hits;

  const texts: string[] = [];
  const slots: Array<{ hit: number; field: "title" | "snippet" }> = [];

  hits.forEach((hit, hitIdx) => {
    if (needsHebrewTranslation(hit.title)) {
      slots.push({ hit: hitIdx, field: "title" });
      texts.push(hit.title);
    }
    if (needsHebrewTranslation(hit.snippet)) {
      slots.push({ hit: hitIdx, field: "snippet" });
      texts.push(hit.snippet!);
    }
  });
  if (!texts.length) return hits;

  try {
    const { texts: translated } = await translateTexts(texts, "he", "en");
    const out = hits.map((h) => ({ ...h }));
    slots.forEach((slot, i) => {
      const value = translated[i]?.trim();
      if (!value) return;
      if (slot.field === "title") out[slot.hit]!.title = value;
      else out[slot.hit]!.snippet = value;
    });
    return out;
  } catch {
    return hits;
  }
}
