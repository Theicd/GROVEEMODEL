import type { UnifiedSearchHit } from "../searchResults/types";
import { radioToSearchHit } from "./adapters";
import { radioQualityScore } from "./ranking";
import type { RadioStation } from "./types";

/** Every N quad tile rotations, inject a regional radio card (TV indices unchanged). */
export const RADIO_INTERSTITIAL_EVERY = 3;

const REGION_ALIASES: Record<string, string[]> = {
  il: ["il", "isr", "israel"],
  de: ["de", "deu", "ger", "germany"],
  us: ["us", "usa", "united states"],
  gb: ["gb", "uk", "gbr", "united kingdom"],
  fr: ["fr", "fra", "france"],
};

function regionAliases(countryCode: string): string[] {
  const cc = countryCode.trim().toLowerCase();
  if (!cc) return [];
  return REGION_ALIASES[cc] ?? [cc];
}

export function radioMatchesRegion(station: RadioStation, countryCode: string): boolean {
  const aliases = regionAliases(countryCode);
  if (!aliases.length) return false;
  const code = (station.countrycode || "").trim().toLowerCase();
  const name = (station.country || "").trim().toLowerCase();
  return aliases.some((a) => code === a || name === a || name.includes(a));
}

function isPlayable(station: RadioStation): boolean {
  return Boolean(station.stream) && station.status !== "offline";
}

/** Regional lineup: favorites first, then user's country, then global top quality. */
export function buildRegionalRadioLineup(
  stations: RadioStation[],
  countryCode: string,
  limit = 24,
): UnifiedSearchHit[] {
  const playable = stations.filter(isPlayable);
  const favorites = playable
    .filter((s) => s.favorite)
    .sort((a, b) => radioQualityScore(b) - radioQualityScore(a));
  const regional = playable
    .filter((s) => !s.favorite && radioMatchesRegion(s, countryCode))
    .sort((a, b) => radioQualityScore(b) - radioQualityScore(a));
  const seen = new Set<string>();
  const picked: RadioStation[] = [];

  const push = (s: RadioStation) => {
    if (seen.has(s.id)) return;
    seen.add(s.id);
    picked.push(s);
  };

  for (const s of favorites) {
    if (picked.length >= limit) break;
    push(s);
  }
  for (const s of regional) {
    if (picked.length >= limit) break;
    push(s);
  }
  if (picked.length < limit && regional.length === 0) {
    const global = playable
      .filter((s) => !seen.has(s.id))
      .sort((a, b) => radioQualityScore(b) - radioQualityScore(a));
    for (const s of global) {
      if (picked.length >= limit) break;
      push(s);
    }
  }

  return picked.map((r) => radioToSearchHit(r));
}

export function isRadioCablePage(pageIndex: number, tvTotal: number): boolean {
  return tvTotal > 0 && pageIndex > tvTotal;
}

export function maxCablePageIndexTvRadio(tvTotal: number, radioTotal: number): number {
  if (tvTotal <= 0) return Math.max(0, radioTotal);
  return tvTotal + Math.max(0, radioTotal);
}

export function radioCablePageIndex(pageIndex: number, tvTotal: number): number {
  return pageIndex - tvTotal - 1;
}

export function nextCablePageWithRadio(
  current: number,
  delta: 1 | -1,
  tvTotal: number,
  radioTotal: number,
): number {
  const max = maxCablePageIndexTvRadio(tvTotal, radioTotal);
  let next = current + delta;
  if (next > max) next = 0;
  if (next < 0) next = max;
  return next;
}

export function firstRadioCablePage(tvTotal: number): number {
  return tvTotal + 1;
}
