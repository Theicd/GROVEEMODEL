import { dbGetUserPrefs, dbPutUserPrefs } from "./indexeddb";
import type { Channel, RadioStation } from "./types";
import { collectDefaultBlacklistIds } from "./defaultBlacklist";
import { channelPassesHeEnCatalog, isPlutoTvChannel, radioPassesHeEnCatalog } from "./heEnCatalogFilter";
import {
  ALL_USER_CATEGORIES,
  defaultBroadcastLanguageForChannel,
  isIsraeliChannel,
  normalizeChannelImageUrl,
  normalizeChannelStreamUrl,
  type ChannelUserOverride,
  type UserChannelCategory,
  type ViewLanguageCode,
} from "./channelUserTaxonomy";

export const PREFS_LOCAL_KEY = "grovee-live-media-user-prefs-v2";

export type LiveMediaUserPrefs = {
  version: 1 | 2;
  favoriteChannelIds: string[];
  favoriteRadioIds: string[];
  blacklistChannelIds: string[];
  blacklistRadioIds: string[];
  /** Per-channel display name, category, broadcast language. */
  channelOverrides?: Record<string, ChannelUserOverride>;
  /** Categories included in cable tuner + TV guide filter. Empty = all. */
  tunerEnabledCategories?: UserChannelCategory[];
  /** Languages shown in tuner + guide. Empty = geo default. */
  viewLanguages?: ViewLanguageCode[];
  /** Browse order for category groups in tuner, favorites, and TV guide. */
  tunerCategoryOrder?: UserChannelCategory[];
  /** One-time seed of default blacklist patterns. */
  defaultBlacklistApplied?: boolean;
  /** Increment when default blacklist rules expand (e.g. Spanish filter v2). */
  defaultBlacklistVersion?: number;
  updatedAt: number;
};

export function emptyUserPrefs(): LiveMediaUserPrefs {
  return {
    version: 2,
    favoriteChannelIds: [],
    favoriteRadioIds: [],
    blacklistChannelIds: [],
    blacklistRadioIds: [],
    channelOverrides: {},
    tunerEnabledCategories: [...ALL_USER_CATEGORIES],
    tunerCategoryOrder: [...ALL_USER_CATEGORIES],
    viewLanguages: [],
    updatedAt: Date.now(),
  };
}

function normalizeCategoryOrder(raw: UserChannelCategory[] | undefined): UserChannelCategory[] {
  const valid = (raw ?? []).filter((c) => ALL_USER_CATEGORIES.includes(c));
  const missing = ALL_USER_CATEGORIES.filter((c) => !valid.includes(c));
  return valid.length ? [...valid, ...missing] : [...ALL_USER_CATEGORIES];
}

function normalizePrefs(parsed: LiveMediaUserPrefs): LiveMediaUserPrefs {
  return {
    version: 2,
    favoriteChannelIds: [...new Set(parsed.favoriteChannelIds ?? [])],
    favoriteRadioIds: [...new Set(parsed.favoriteRadioIds ?? [])],
    blacklistChannelIds: [...new Set(parsed.blacklistChannelIds ?? [])],
    blacklistRadioIds: [...new Set(parsed.blacklistRadioIds ?? [])],
    channelOverrides: { ...(parsed.channelOverrides ?? {}) },
    tunerEnabledCategories:
      parsed.tunerEnabledCategories?.filter((c) => ALL_USER_CATEGORIES.includes(c)) ?? [...ALL_USER_CATEGORIES],
    tunerCategoryOrder: normalizeCategoryOrder(parsed.tunerCategoryOrder),
    viewLanguages: [...new Set(parsed.viewLanguages ?? [])],
    defaultBlacklistApplied: parsed.defaultBlacklistApplied,
    defaultBlacklistVersion: parsed.defaultBlacklistVersion,
    updatedAt: parsed.updatedAt ?? Date.now(),
  };
}

function readLocalPrefs(): LiveMediaUserPrefs | null {
  try {
    let raw = localStorage.getItem(PREFS_LOCAL_KEY);
    if (!raw) {
      raw = localStorage.getItem("grovee-live-media-user-prefs-v1");
    }
    if (!raw) return null;
    return normalizePrefs(JSON.parse(raw) as LiveMediaUserPrefs);
  } catch {
    return null;
  }
}

function writeLocalPrefs(prefs: LiveMediaUserPrefs): void {
  try {
    localStorage.setItem(PREFS_LOCAL_KEY, JSON.stringify(prefs));
  } catch {
    /* quota / private mode */
  }
}

export async function loadUserPrefs(): Promise<LiveMediaUserPrefs> {
  let prefs = await dbGetUserPrefs();
  if (!prefs) prefs = readLocalPrefs();
  if (!prefs) prefs = emptyUserPrefs();
  prefs = normalizePrefs(prefs);
  writeLocalPrefs(prefs);
  return prefs;
}

/** Recover stars saved on channel/radio rows before userPrefs existed. */
export async function migrateLegacyFavoritesIntoPrefs(
  channels: Channel[],
  radio: RadioStation[],
  prefs: LiveMediaUserPrefs,
): Promise<LiveMediaUserPrefs> {
  const legacyTv = channels.filter((c) => c.favorite).map((c) => c.id);
  const legacyRadio = radio.filter((r) => r.favorite).map((r) => r.id);
  const favoriteChannelIds = [...new Set([...prefs.favoriteChannelIds, ...legacyTv])];
  const favoriteRadioIds = [...new Set([...prefs.favoriteRadioIds, ...legacyRadio])];
  const blacklistChannelIds = prefs.blacklistChannelIds.filter((id) => !favoriteChannelIds.includes(id));
  const blacklistRadioIds = prefs.blacklistRadioIds.filter((id) => !favoriteRadioIds.includes(id));
  const changed =
    favoriteChannelIds.length !== prefs.favoriteChannelIds.length ||
    favoriteRadioIds.length !== prefs.favoriteRadioIds.length ||
    blacklistChannelIds.length !== prefs.blacklistChannelIds.length ||
    blacklistRadioIds.length !== prefs.blacklistRadioIds.length;
  if (!changed) return prefs;
  const next: LiveMediaUserPrefs = {
    ...prefs,
    favoriteChannelIds,
    favoriteRadioIds,
    blacklistChannelIds,
    blacklistRadioIds,
  };
  await saveUserPrefs(next);
  return next;
}

export async function saveUserPrefs(prefs: LiveMediaUserPrefs): Promise<void> {
  const next = { ...prefs, updatedAt: Date.now() };
  await dbPutUserPrefs(next);
  writeLocalPrefs(next);
}

export function exportUserPrefsJson(prefs: LiveMediaUserPrefs): string {
  return JSON.stringify(prefs, null, 2);
}

/** Favorites must never stay hidden behind the blacklist. */
export function releaseBlacklistedFavorites(prefs: LiveMediaUserPrefs): LiveMediaUserPrefs {
  const favTv = favoriteChannelSet(prefs);
  const favRadio = favoriteRadioSet(prefs);
  const blacklistChannelIds = prefs.blacklistChannelIds.filter((id) => !favTv.has(id));
  const blacklistRadioIds = prefs.blacklistRadioIds.filter((id) => !favRadio.has(id));
  if (
    blacklistChannelIds.length === prefs.blacklistChannelIds.length &&
    blacklistRadioIds.length === prefs.blacklistRadioIds.length
  ) {
    return prefs;
  }
  return { ...prefs, blacklistChannelIds, blacklistRadioIds };
}

export function importUserPrefsJson(raw: string): LiveMediaUserPrefs {
  const parsed = JSON.parse(raw) as LiveMediaUserPrefs;
  if (parsed.version !== 1 && parsed.version !== 2) throw new Error("Unsupported prefs version");
  return normalizePrefs({ ...parsed, updatedAt: Date.now() });
}

export async function setChannelOverride(
  channelId: string,
  patch: ChannelUserOverride | null,
): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const overrides = { ...(prefs.channelOverrides ?? {}) };
  if (!patch) {
    delete overrides[channelId];
  } else {
    const prev = overrides[channelId] ?? {};
    const next: ChannelUserOverride = { ...prev };
    if ("displayName" in patch) {
      const dn = patch.displayName?.trim();
      if (dn) next.displayName = dn;
      else delete next.displayName;
    }
    if (patch.category !== undefined) next.category = patch.category;
    if ("broadcastLanguage" in patch) {
      if (patch.broadcastLanguage) next.broadcastLanguage = patch.broadcastLanguage;
      else delete next.broadcastLanguage;
    }
    if ("imageUrl" in patch) {
      const img = normalizeChannelImageUrl(patch.imageUrl);
      if (img) next.imageUrl = img;
      else delete next.imageUrl;
    }
    if ("streamUrl" in patch) {
      const stream = normalizeChannelStreamUrl(patch.streamUrl);
      if (stream) next.streamUrl = stream;
      else delete next.streamUrl;
    }
    if (!next.displayName && !next.category && !next.broadcastLanguage && !next.imageUrl && !next.streamUrl) {
      delete overrides[channelId];
    } else {
      overrides[channelId] = next;
    }
  }
  const next = { ...prefs, channelOverrides: overrides };
  await saveUserPrefs(next);
  return next;
}

/** Remove Hebrew language overrides mistakenly saved on international channels. */
export function sanitizeStaleLanguageOverrides(
  channels: Channel[],
  prefs: LiveMediaUserPrefs,
): { prefs: LiveMediaUserPrefs; changed: boolean } {
  const overrides = prefs.channelOverrides ?? {};
  let changed = false;
  const nextOverrides: Record<string, ChannelUserOverride> = { ...overrides };

  for (const [id, o] of Object.entries(overrides)) {
    if (o.broadcastLanguage !== "heb") continue;
    const c = channels.find((ch) => ch.id === id);
    if (!c || isIsraeliChannel(c)) continue;
    if (defaultBroadcastLanguageForChannel(c) === "heb") continue;
    const copy = { ...o };
    delete copy.broadcastLanguage;
    changed = true;
    if (!copy.displayName && !copy.category && !copy.imageUrl && !copy.streamUrl) {
      delete nextOverrides[id];
    } else {
      nextOverrides[id] = copy;
    }
  }

  if (!changed) return { prefs, changed: false };
  return {
    prefs: { ...prefs, channelOverrides: nextOverrides, updatedAt: Date.now() },
    changed: true,
  };
}

export async function updateTunerPreferences(patch: {
  tunerEnabledCategories?: UserChannelCategory[];
  tunerCategoryOrder?: UserChannelCategory[];
  viewLanguages?: ViewLanguageCode[];
}): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const next = {
    ...prefs,
    ...(patch.tunerEnabledCategories !== undefined
      ? { tunerEnabledCategories: [...patch.tunerEnabledCategories] }
      : {}),
    ...(patch.tunerCategoryOrder !== undefined
      ? { tunerCategoryOrder: normalizeCategoryOrder(patch.tunerCategoryOrder) }
      : {}),
    ...(patch.viewLanguages !== undefined ? { viewLanguages: [...patch.viewLanguages] } : {}),
  };
  await saveUserPrefs(next);
  return next;
}

export function blacklistChannelSet(prefs: LiveMediaUserPrefs): Set<string> {
  return new Set(prefs.blacklistChannelIds);
}

export function blacklistRadioSet(prefs: LiveMediaUserPrefs): Set<string> {
  return new Set(prefs.blacklistRadioIds);
}

export function favoriteChannelSet(prefs: LiveMediaUserPrefs): Set<string> {
  return new Set(prefs.favoriteChannelIds);
}

export function favoriteRadioSet(prefs: LiveMediaUserPrefs): Set<string> {
  return new Set(prefs.favoriteRadioIds);
}

export const DEFAULT_BLACKLIST_VERSION = 4;

function resolvedBlacklistVersion(prefs: LiveMediaUserPrefs): number {
  if (prefs.defaultBlacklistVersion != null) return prefs.defaultBlacklistVersion;
  return prefs.defaultBlacklistApplied ? 1 : 0;
}

export async function applyDefaultBlacklistOnce(
  channels: Channel[],
  radio: RadioStation[],
  prefs: LiveMediaUserPrefs,
): Promise<LiveMediaUserPrefs> {
  if (resolvedBlacklistVersion(prefs) >= DEFAULT_BLACKLIST_VERSION) return prefs;
  const favTv = favoriteChannelSet(prefs);
  const favRadio = favoriteRadioSet(prefs);
  const { channelIds, radioIds } = collectDefaultBlacklistIds(channels, radio);
  const next: LiveMediaUserPrefs = {
    ...prefs,
    blacklistChannelIds: [
      ...new Set([...prefs.blacklistChannelIds, ...channelIds.filter((id) => !favTv.has(id))]),
    ],
    blacklistRadioIds: [
      ...new Set([...prefs.blacklistRadioIds, ...radioIds.filter((id) => !favRadio.has(id))]),
    ],
    defaultBlacklistApplied: true,
    defaultBlacklistVersion: DEFAULT_BLACKLIST_VERSION,
    updatedAt: Date.now(),
  };
  await saveUserPrefs(next);
  return next;
}

export function applyPrefsToChannels(channels: Channel[], prefs: LiveMediaUserPrefs): Channel[] {
  const fav = favoriteChannelSet(prefs);
  return channels.map((c) => ({ ...c, favorite: fav.has(c.id) }));
}

export function applyPrefsToRadio(radio: RadioStation[], prefs: LiveMediaUserPrefs): RadioStation[] {
  const fav = favoriteRadioSet(prefs);
  return radio.map((r) => ({ ...r, favorite: fav.has(r.id) }));
}

export function visibleChannels(channels: Channel[], prefs: LiveMediaUserPrefs): Channel[] {
  const blocked = blacklistChannelSet(prefs);
  const fav = favoriteChannelSet(prefs);
  return channels.filter((c) => {
    if (isPlutoTvChannel(c)) return false;
    if (blocked.has(c.id) && !fav.has(c.id)) return false;
    if (!fav.has(c.id) && !channelPassesHeEnCatalog(c)) return false;
    return true;
  });
}

export function visibleRadio(radio: RadioStation[], prefs: LiveMediaUserPrefs): RadioStation[] {
  const blocked = blacklistRadioSet(prefs);
  const fav = favoriteRadioSet(prefs);
  return radio.filter((r) => {
    if (blocked.has(r.id) && !fav.has(r.id)) return false;
    if (!fav.has(r.id) && !radioPassesHeEnCatalog(r)) return false;
    return true;
  });
}

export async function setChannelFavorite(channelId: string, favorite: boolean): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const set = new Set(prefs.favoriteChannelIds);
  if (favorite) set.add(channelId);
  else set.delete(channelId);
  const next = { ...prefs, favoriteChannelIds: [...set] };
  await saveUserPrefs(next);
  return next;
}

export async function setRadioFavorite(stationId: string, favorite: boolean): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const set = new Set(prefs.favoriteRadioIds);
  if (favorite) set.add(stationId);
  else set.delete(stationId);
  const next = { ...prefs, favoriteRadioIds: [...set] };
  await saveUserPrefs(next);
  return next;
}

export async function blacklistChannel(channelId: string): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const blocked = new Set(prefs.blacklistChannelIds);
  const fav = new Set(prefs.favoriteChannelIds);
  blocked.add(channelId);
  fav.delete(channelId);
  const next: LiveMediaUserPrefs = {
    ...prefs,
    blacklistChannelIds: [...blocked],
    favoriteChannelIds: [...fav],
  };
  await saveUserPrefs(next);
  return next;
}

export async function blacklistRadio(stationId: string): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const blocked = new Set(prefs.blacklistRadioIds);
  const fav = new Set(prefs.favoriteRadioIds);
  blocked.add(stationId);
  fav.delete(stationId);
  const next: LiveMediaUserPrefs = {
    ...prefs,
    blacklistRadioIds: [...blocked],
    favoriteRadioIds: [...fav],
  };
  await saveUserPrefs(next);
  return next;
}

export async function restoreChannelFromBlacklist(channelId: string): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const next = {
    ...prefs,
    blacklistChannelIds: prefs.blacklistChannelIds.filter((id) => id !== channelId),
  };
  await saveUserPrefs(next);
  return next;
}

export async function restoreRadioFromBlacklist(stationId: string): Promise<LiveMediaUserPrefs> {
  const prefs = await loadUserPrefs();
  const next = {
    ...prefs,
    blacklistRadioIds: prefs.blacklistRadioIds.filter((id) => id !== stationId),
  };
  await saveUserPrefs(next);
  return next;
}
