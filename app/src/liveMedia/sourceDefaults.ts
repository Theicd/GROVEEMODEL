import type { Source } from "./types";

/** Infer default country/language from iptv-org M3U URL paths. */
export function inferM3UParseDefaults(source: Source): {
  defaultCountry?: string;
  defaultLanguage?: string;
} {
  const url = source.url.toLowerCase();
  const country = url.match(/\/countries\/([a-z]{2})\.m3u/)?.[1];
  const language = url.match(/\/languages\/([a-z]{3})\.m3u/)?.[1];
  return {
    defaultCountry: country,
    defaultLanguage: language,
  };
}

/** Sync Israel + Hebrew feeds before large category lists. */
export function liveMediaSourceSortPriority(source: Source): number {
  if (source.id === "iptv-org-il" || source.url.includes("/countries/il.m3u")) return 0;
  if (source.id.includes("heb") || source.url.includes("/languages/heb")) return 1;
  if (source.type === "radio" && source.url.includes("israel")) return 2;
  if (source.type === "radio") return 5;
  if (source.url.includes("/categories/")) return 8;
  return 4;
}
