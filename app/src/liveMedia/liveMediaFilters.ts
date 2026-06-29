import type { UnifiedSearchHit } from "../searchResults/types";
import { hitChannelId } from "./channelDisplay";
import {
  ALL_USER_CATEGORIES,
  defaultViewLanguagesForCountry,
  normalizeViewLanguageCode,
  resolveBroadcastLanguage,
  resolveUserCategory,
  type UserChannelCategory,
  type ViewLanguageCode,
} from "./channelUserTaxonomy";
import type { Channel } from "./types";
import type { LiveMediaUserPrefs } from "./userPrefs";

export function effectiveViewLanguages(prefs: LiveMediaUserPrefs, geoCountry = ""): ViewLanguageCode[] {
  const raw = prefs.viewLanguages ?? [];
  const normalized = raw
    .map((c) => normalizeViewLanguageCode(c))
    .filter((c): c is ViewLanguageCode => c != null && c !== "und");
  if (normalized.length) return [...new Set(normalized)];
  return defaultViewLanguagesForCountry(geoCountry || "us");
}

export function effectiveTunerCategories(prefs: LiveMediaUserPrefs): UserChannelCategory[] {
  const cats = prefs.tunerEnabledCategories ?? [];
  const valid = cats.filter((c) => ALL_USER_CATEGORIES.includes(c));
  return valid.length ? valid : [...ALL_USER_CATEGORIES];
}

/** User-defined browse order; unknown categories append at the end. */
export function effectiveCategoryOrder(prefs: LiveMediaUserPrefs | null | undefined): UserChannelCategory[] {
  if (!prefs) return [...ALL_USER_CATEGORIES];
  const stored = prefs.tunerCategoryOrder ?? [];
  const valid = stored.filter((c) => ALL_USER_CATEGORIES.includes(c));
  const missing = ALL_USER_CATEGORIES.filter((c) => !valid.includes(c));
  return valid.length ? [...valid, ...missing] : [...ALL_USER_CATEGORIES];
}

function channelById(channels: Channel[], id: string): Channel | undefined {
  return channels.find((c) => c.id === id);
}

export function channelMatchesViewLanguages(
  c: Channel,
  prefs: LiveMediaUserPrefs,
  viewLangs: ViewLanguageCode[],
): boolean {
  const override = prefs.channelOverrides?.[c.id];
  const lang = resolveBroadcastLanguage(c, override);
  if (lang === "und") return true;
  return viewLangs.includes(lang);
}

export function channelMatchesTunerCategories(
  c: Channel,
  prefs: LiveMediaUserPrefs,
  categories: UserChannelCategory[],
): boolean {
  const cat = resolveUserCategory(c.id, c, prefs.channelOverrides);
  return categories.includes(cat);
}

/** Filter favorite TV hits for cable tuner / guide (category + language). */
export function filterTunerFavorites(
  hits: UnifiedSearchHit[],
  channels: Channel[],
  prefs: LiveMediaUserPrefs | null | undefined,
  geoCountry = "",
): UnifiedSearchHit[] {
  if (!prefs) return hits;
  const viewLangs = effectiveViewLanguages(prefs, geoCountry);
  const tunerCats = effectiveTunerCategories(prefs);

  return hits.filter((hit) => {
    if (hit.kind !== "livetv") return true;
    const channelId = hitChannelId(hit);
    if (!channelId) return true;
    const c = channelById(channels, channelId);
    if (!c) return true;
    return (
      channelMatchesTunerCategories(c, prefs, tunerCats) &&
      channelMatchesViewLanguages(c, prefs, viewLangs)
    );
  });
}

export function groupHitsByUserCategory(
  hits: UnifiedSearchHit[],
  channels: Channel[],
  prefs: LiveMediaUserPrefs | null | undefined,
): Map<UserChannelCategory, UnifiedSearchHit[]> {
  const map = new Map<UserChannelCategory, UnifiedSearchHit[]>();
  for (const cat of ALL_USER_CATEGORIES) map.set(cat, []);

  for (const hit of hits) {
    if (hit.kind !== "livetv") continue;
    const channelId = hitChannelId(hit);
    if (!channelId) continue;
    const c = channelById(channels, channelId);
    if (!c) continue;
    const cat =
      (hit.meta?.userCategory as UserChannelCategory | undefined) ??
      resolveUserCategory(channelId, c, prefs?.channelOverrides);
    const list = map.get(cat) ?? [];
    list.push(hit);
    map.set(cat, list);
  }
  return map;
}

/** Flatten hits in user category order (tuner channel-up/down). */
export function sortHitsByCategoryOrder(
  hits: UnifiedSearchHit[],
  channels: Channel[],
  prefs: LiveMediaUserPrefs | null | undefined,
): UnifiedSearchHit[] {
  const order = effectiveCategoryOrder(prefs);
  const grouped = groupHitsByUserCategory(hits, channels, prefs);
  const out: UnifiedSearchHit[] = [];
  for (const cat of order) {
    out.push(...(grouped.get(cat) ?? []));
  }
  return out;
}
