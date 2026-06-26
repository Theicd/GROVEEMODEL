import type { ChatUiLanguage } from "../ui/useUiLanguage";

export type TmdbLanguage = "he-IL" | "en-US";

export function tmdbLocaleForUi(lang: ChatUiLanguage): TmdbLanguage {
  return lang === "he" ? "he-IL" : "en-US";
}

/** English fallback when a localized TMDB field is missing. */
export function tmdbFallbackLocale(lang: TmdbLanguage): TmdbLanguage | null {
  return lang === "he-IL" ? "en-US" : null;
}
