import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { TmdbLanguage } from "./tmdbLocale";

/** Text is Latin-only (no Hebrew) — worth auto-translating for Hebrew UI. */
export function needsHebrewTranslation(text: string | null | undefined): boolean {
  const t = text?.trim();
  if (!t) return false;
  if (/[\u0590-\u05ff]/.test(t)) return false;
  return /[a-z]/i.test(t);
}

/** Fill missing Hebrew from Google Translate when TMDB has no he-IL copy. */
export async function localizeTmdbMetaForUi<T extends { title: string; overview: string | null }>(
  meta: T,
  language: TmdbLanguage,
): Promise<T> {
  if (language !== "he-IL") return meta;

  const pending: Array<"title" | "overview"> = [];
  const texts: string[] = [];
  if (needsHebrewTranslation(meta.title)) {
    pending.push("title");
    texts.push(meta.title);
  }
  if (needsHebrewTranslation(meta.overview)) {
    pending.push("overview");
    texts.push(meta.overview!);
  }
  if (!texts.length) return meta;

  try {
    const { texts: translated } = await translateTexts(texts, "he", "en");
    let idx = 0;
    let title = meta.title;
    let overview = meta.overview;
    for (const field of pending) {
      const value = translated[idx++]?.trim();
      if (!value) continue;
      if (field === "title") title = value;
      else overview = value;
    }
    return { ...meta, title, overview };
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
