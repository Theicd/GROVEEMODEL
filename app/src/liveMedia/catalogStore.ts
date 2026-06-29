import {
  dbClearChannelsBySource,
  dbGetAllChannels,
  dbGetAllRadio,
  dbGetAllSources,
  dbGetStats,
  dbPutChannels,
  dbPutRadio,
  dbPutSource,
  dbUpdateChannel,
  dbUpdateRadio,
} from "./indexeddb";
import { parseM3U } from "./m3u-parser";
import { parseRadioStations } from "./radio-parser";
import { BUILTIN_SOURCES } from "./sources";
import { fetchCatalogText } from "./fetchCatalog";
import { inferM3UParseDefaults, liveMediaSourceSortPriority } from "./sourceDefaults";
import { enrichChannel, enrichRadio } from "./languageMetadata";
import {
  migrateLegacyFavoritesIntoPrefs,
  releaseBlacklistedFavorites,
  applyDefaultBlacklistOnce,
  applyPrefsToChannels,
  applyPrefsToRadio,
  blacklistChannel,
  blacklistChannelSet,
  blacklistRadio,
  blacklistRadioSet,
  exportUserPrefsJson,
  importUserPrefsJson,
  loadUserPrefs,
  restoreChannelFromBlacklist,
  restoreRadioFromBlacklist,
  sanitizeStaleLanguageOverrides,
  saveUserPrefs,
  setChannelOverride,
  setChannelFavorite,
  setRadioFavorite,
  updateTunerPreferences,
  visibleChannels,
  visibleRadio,
  type LiveMediaUserPrefs,
} from "./userPrefs";
import { channelQualityScore, radioQualityScore } from "./ranking";
import type { ChannelUserOverride } from "./channelUserTaxonomy";
import {
  fetchCuratedFavoritesFromRepo,
  injectCuratedChannels,
  mergeCuratedFavoritesIntoPrefs,
  persistCuratedFavoritesToRepo,
} from "./curatedFavorites";
import type { LiveMediaCatalogSummary } from "./runtimeState";
import {
  categoryBreakdown,
  notifyLiveMediaSummary,
  resetLiveMediaProgress,
  setLiveMediaError,
  setLiveMediaProgress,
  statusBreakdown,
} from "./runtimeState";
import { validateRadioStreamWithMetrics, validateStreamWithMetrics } from "./validator";
import type { Channel, RadioStation, Source } from "./types";

let syncPromise: Promise<void> | null = null;
let validatePromise: Promise<void> | null = null;
let memoryChannels: Channel[] | null = null;
let memoryRadio: RadioStation[] | null = null;
let memoryPrefs: LiveMediaUserPrefs | null = null;

async function mergeRepoCuratedFavorites(
  prefs: LiveMediaUserPrefs,
): Promise<LiveMediaUserPrefs> {
  const curated = await fetchCuratedFavoritesFromRepo();
  const { prefs: merged, changed } = mergeCuratedFavoritesIntoPrefs(prefs, curated);
  if (!changed) return prefs;
  await saveUserPrefs(merged);
  return merged;
}

async function pushLocalFavoritesToRepoIfNeeded(
  channels: Channel[],
  radio: RadioStation[],
  prefs: LiveMediaUserPrefs,
): Promise<void> {
  if (!import.meta.env.DEV) return;
  if (prefs.favoriteChannelIds.length === 0 && prefs.favoriteRadioIds.length === 0) return;

  const curated = await fetchCuratedFavoritesFromRepo();
  const repoIds = new Set([
    ...(curated?.channels.map((c) => c.id) ?? []),
    ...(curated?.radio.map((r) => r.id) ?? []),
  ]);
  const hasUnsynced = [...prefs.favoriteChannelIds, ...prefs.favoriteRadioIds].some((id) => !repoIds.has(id));
  if (!hasUnsynced) return;
  void persistCuratedFavoritesToRepo(channels, radio, prefs);
}

async function syncFavoritesToRepo(
  channels: Channel[],
  radio: RadioStation[],
  prefs: LiveMediaUserPrefs,
): Promise<void> {
  void persistCuratedFavoritesToRepo(channels, radio, prefs);
}

async function ensureBuiltinSources(): Promise<Source[]> {
  const existing = await dbGetAllSources();
  const byId = new Map(existing.map((s) => [s.id, s]));
  for (const bs of BUILTIN_SOURCES) {
    if (!byId.has(bs.id)) {
      await dbPutSource(bs);
      byId.set(bs.id, bs);
    }
  }
  return [...byId.values()];
}

export async function buildLiveMediaCatalogSummary(): Promise<LiveMediaCatalogSummary> {
  await ensureBuiltinSources();
  const [channels, radio, sources] = await Promise.all([
    memoryChannels ?? dbGetAllChannels(),
    memoryRadio ?? dbGetAllRadio(),
    dbGetAllSources(),
  ]);
  const lastSyncAt = sources.reduce<number | null>((max, s) => {
    if (!s.lastSync) return max;
    return max == null || s.lastSync > max ? s.lastSync : max;
  }, null);
  const { getLiveMediaProgress, getLiveMediaLastError } = await import("./runtimeState");
  return {
    channels: channels.length,
    radio: radio.length,
    channelStatus: statusBreakdown(channels),
    radioStatus: statusBreakdown(radio),
    categories: categoryBreakdown(channels),
    sources,
    lastSyncAt,
    progress: getLiveMediaProgress(),
    lastError: getLiveMediaLastError(),
  };
}

function mergeChannelMeta(incoming: Channel, old?: Channel, prefs?: LiveMediaUserPrefs): Channel {
  const enriched = enrichChannel(incoming);
  if (!old) {
    const favorite =
      prefs?.favoriteChannelIds.includes(enriched.id) || enriched.favorite ? true : enriched.favorite;
    return { ...enriched, favorite, qualityScore: channelQualityScore({ ...enriched, favorite }) };
  }
  const merged: Channel = {
    ...enriched,
    favorite:
      prefs?.favoriteChannelIds.includes(enriched.id) || old.favorite
        ? true
        : false,
    country: enriched.country || old.country,
    language: enriched.language || old.language,
    languages: enriched.languages?.length ? enriched.languages : old.languages,
    loadTimeMs: old.loadTimeMs ?? enriched.loadTimeMs,
    status:
      old.status === "working" && enriched.status === "unknown"
        ? old.status
        : enriched.status ?? old.status,
    lastCheck: Math.max(old.lastCheck ?? 0, enriched.lastCheck ?? 0) || old.lastCheck,
  };
  merged.qualityScore = channelQualityScore(merged);
  return merged;
}

function mergeRadioMeta(incoming: RadioStation, old?: RadioStation, prefs?: LiveMediaUserPrefs): RadioStation {
  const enriched = enrichRadio(incoming);
  if (!old) {
    const favorite =
      prefs?.favoriteRadioIds.includes(enriched.id) || enriched.favorite ? true : enriched.favorite;
    return { ...enriched, favorite, qualityScore: radioQualityScore({ ...enriched, favorite }) };
  }
  const merged: RadioStation = {
    ...enriched,
    favorite:
      prefs?.favoriteRadioIds.includes(enriched.id) || old.favorite
        ? true
        : false,
    languages: enriched.languages?.length ? enriched.languages : old.languages,
    language: enriched.language || old.language,
    loadTimeMs: old.loadTimeMs ?? enriched.loadTimeMs,
    status:
      old.status === "working" && enriched.status === "unknown"
        ? old.status
        : enriched.status ?? old.status,
    lastCheck: Math.max(old.lastCheck ?? 0, enriched.lastCheck ?? 0) || old.lastCheck,
  };
  merged.qualityScore = radioQualityScore(merged);
  return merged;
}

export async function syncLiveMediaSource(source: Source): Promise<number> {
  const text = await fetchCatalogText(source.url);
  const parseDefaults = inferM3UParseDefaults(source);
  const prefs = memoryPrefs ?? (await loadUserPrefs());
  memoryPrefs = prefs;
  const blockedTv = blacklistChannelSet(prefs);
  const blockedRadio = blacklistRadioSet(prefs);

  if (source.type === "iptv") {
    const prev = await dbGetAllChannels();
    const prevById = new Map(prev.map((c) => [c.id, c]));
    await dbClearChannelsBySource(source.id);
    const channels = parseM3U(text, { source: source.id, ...parseDefaults });
    const merged = channels
      .filter((c) => !blockedTv.has(c.id))
      .map((c) => mergeChannelMeta(c, prevById.get(c.id), prefs));
    await dbPutChannels(merged);
    memoryChannels = null;
    await dbPutSource({ ...source, lastSync: Date.now(), channelCount: merged.length });
    return merged.length;
  }
  if (source.type === "radio") {
    const data = JSON.parse(text) as unknown[];
    const stations = parseRadioStations(data as Parameters<typeof parseRadioStations>[0]);
    const prev = await dbGetAllRadio();
    const prevById = new Map(prev.map((r) => [r.id, r]));
    const merged = stations
      .filter((s) => !blockedRadio.has(s.id))
      .map((s) => mergeRadioMeta(s, prevById.get(s.id), prefs));
    await dbPutRadio(merged);
    memoryRadio = null;
    await dbPutSource({ ...source, lastSync: Date.now(), channelCount: merged.length });
    return merged.length;
  }
  return 0;
}

export async function syncAllLiveMediaSources(): Promise<{ channels: number; radio: number }> {
  if (syncPromise) {
    await syncPromise;
    const stats = await dbGetStats();
    return { channels: stats.channels, radio: stats.radio };
  }

  syncPromise = (async () => {
    setLiveMediaError(null);
    const sources = (await ensureBuiltinSources())
      .filter((s) => s.enabled)
      .sort((a, b) => liveMediaSourceSortPriority(a) - liveMediaSourceSortPriority(b));
    setLiveMediaProgress({
      phase: "syncing",
      current: 0,
      total: sources.length,
      label: sources[0]?.name ?? "",
    });
    notifyLiveMediaSummary();

    for (let i = 0; i < sources.length; i++) {
      const source = sources[i];
      setLiveMediaProgress({
        phase: "syncing",
        current: i,
        total: sources.length,
        label: source.name,
      });
      notifyLiveMediaSummary();
      try {
        await syncLiveMediaSource(source);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn("[liveMedia] sync failed:", source.name, err);
        setLiveMediaError(`${source.name}: ${msg}`);
      }
      setLiveMediaProgress({
        phase: "syncing",
        current: i + 1,
        total: sources.length,
        label: source.name,
      });
      notifyLiveMediaSummary();
    }

    memoryChannels = null;
    memoryRadio = null;
    resetLiveMediaProgress();
    notifyLiveMediaSummary();
  })().finally(() => {
    syncPromise = null;
  });

  await syncPromise;
  const stats = await dbGetStats();
  return { channels: stats.channels, radio: stats.radio };
}

const VALIDATE_BATCH = 6;
const VALIDATE_TIMEOUT_MS = 10_000;
let validateAbort: AbortController | null = null;

export function cancelLiveMediaValidation(): void {
  validateAbort?.abort();
}

export async function validateAllLiveMediaStreams(options?: {
  radioOnly?: boolean;
  tvOnly?: boolean;
}): Promise<void> {
  if (validatePromise) {
    await validatePromise;
    return;
  }

  validateAbort = new AbortController();
  const signal = validateAbort.signal;

  validatePromise = (async () => {
    setLiveMediaError(null);
    const channels = await dbGetAllChannels();
    const radio = await dbGetAllRadio();
    const needsCheck = (s?: string) => !s || s === "unknown" || s === "warning";

    const radioTargets = options?.tvOnly
      ? []
      : radio.filter((r) => needsCheck(r.status));
    const tvTargets = options?.radioOnly
      ? []
      : channels.filter((c) => needsCheck(c.status));

    const total = radioTargets.length + tvTargets.length;
    if (total === 0) return;

    let done = 0;
    const tick = (label: string) => {
      setLiveMediaProgress({
        phase: "validating",
        current: done,
        total,
        label,
      });
      notifyLiveMediaSummary();
    };

    tick(radioTargets.length ? "📻 Radio QA" : "📺 TV QA");

    const runBatch = async <T extends Channel | RadioStation>(
      items: T[],
      kind: "radio" | "tv",
      validate: (item: T) => Promise<void>,
      labelFor: (item: T) => string,
    ) => {
      for (let i = 0; i < items.length; i += VALIDATE_BATCH) {
        if (signal.aborted) return;
        const batch = items.slice(i, i + VALIDATE_BATCH);
        await Promise.all(batch.map((item) => validate(item)));
        done += batch.length;
        tick(`${kind === "radio" ? "📻" : "📺"} ${labelFor(batch[batch.length - 1])}`);
        await new Promise((r) => setTimeout(r, 0));
      }
    };

    await runBatch(
      radioTargets,
      "radio",
      async (st) => {
        const { status, loadTimeMs } = await validateRadioStreamWithMetrics(st.stream, VALIDATE_TIMEOUT_MS);
        const updated = mergeRadioMeta(
          { ...st, status, lastCheck: Date.now(), loadTimeMs },
          st,
        );
        await updateRadioStatus(updated);
      },
      (st) => st.name,
    );

    if (!signal.aborted) {
      await runBatch(
        tvTargets,
        "tv",
        async (ch) => {
          const { status, loadTimeMs } = await validateStreamWithMetrics(ch.stream, VALIDATE_TIMEOUT_MS);
          const updated = mergeChannelMeta(
            { ...ch, status, lastCheck: Date.now(), loadTimeMs },
            ch,
          );
          await updateChannelStatus(updated);
        },
        (ch) => ch.name,
      );
    }

    resetLiveMediaProgress();
    notifyLiveMediaSummary();
  })().finally(() => {
    validatePromise = null;
    validateAbort = null;
  });

  await validatePromise;
}

export async function toggleLiveMediaSource(id: string, enabled: boolean): Promise<void> {
  const sources = await dbGetAllSources();
  const source = sources.find((s) => s.id === id);
  if (!source) return;
  await dbPutSource({ ...source, enabled });
  notifyLiveMediaSummary();
}

/** Load catalog from IndexedDB; sync enabled sources if empty. */
export async function ensureLiveMediaLibrary(): Promise<{
  channels: Channel[];
  radio: RadioStation[];
  prefs: LiveMediaUserPrefs;
}> {
  if (memoryChannels && memoryRadio && memoryPrefs) {
    return {
      channels: visibleChannels(memoryChannels, memoryPrefs),
      radio: visibleRadio(memoryRadio, memoryPrefs),
      prefs: memoryPrefs,
    };
  }
  await ensureBuiltinSources();
  let channels = (await dbGetAllChannels()).map(enrichChannel);
  let radio = (await dbGetAllRadio()).map(enrichRadio);
  const curated = await fetchCuratedFavoritesFromRepo();
  channels = injectCuratedChannels(channels, curated).map(enrichChannel);
  if (channels.length === 0 && radio.length === 0) {
    await syncAllLiveMediaSources();
    channels = injectCuratedChannels((await dbGetAllChannels()).map(enrichChannel), curated).map(enrichChannel);
    radio = (await dbGetAllRadio()).map(enrichRadio);
  }
  let prefs = await loadUserPrefs();
  const mergedPrefs = mergeCuratedFavoritesIntoPrefs(prefs, curated);
  if (mergedPrefs.changed) {
    await saveUserPrefs(mergedPrefs.prefs);
    prefs = mergedPrefs.prefs;
  }
  prefs = await migrateLegacyFavoritesIntoPrefs(channels, radio, prefs);
  const repaired = releaseBlacklistedFavorites(prefs);
  if (
    repaired.blacklistChannelIds.length !== prefs.blacklistChannelIds.length ||
    repaired.blacklistRadioIds.length !== prefs.blacklistRadioIds.length
  ) {
    await saveUserPrefs(repaired);
    prefs = repaired;
  }
  prefs = await applyDefaultBlacklistOnce(channels, radio, prefs);
  const langSanitized = sanitizeStaleLanguageOverrides(channels, prefs);
  if (langSanitized.changed) {
    await saveUserPrefs(langSanitized.prefs);
    prefs = langSanitized.prefs;
  }
  channels = applyPrefsToChannels(channels, prefs);
  radio = applyPrefsToRadio(radio, prefs);
  memoryChannels = channels;
  memoryRadio = radio;
  memoryPrefs = prefs;
  await pushLocalFavoritesToRepoIfNeeded(channels, radio, prefs);
  notifyLiveMediaSummary();
  return {
    channels: visibleChannels(channels, prefs),
    radio: visibleRadio(radio, prefs),
    prefs,
  };
}

export async function updateChannelStatus(channel: Channel): Promise<void> {
  await dbUpdateChannel(channel);
  if (memoryChannels) {
    memoryChannels = memoryChannels.map((c) => (c.id === channel.id ? channel : c));
  }
}

export async function updateRadioStatus(station: RadioStation): Promise<void> {
  await dbUpdateRadio(station);
  if (memoryRadio) {
    memoryRadio = memoryRadio.map((r) => (r.id === station.id ? station : r));
  }
}

export async function toggleChannelFavorite(channelId: string): Promise<boolean> {
  let channels = memoryChannels ?? (await dbGetAllChannels()).map(enrichChannel);
  const ch = channels.find((c) => c.id === channelId);
  if (!ch) return false;
  const nextFav = !ch.favorite;
  memoryPrefs = await setChannelFavorite(channelId, nextFav);
  const updated: Channel = { ...ch, favorite: nextFav };
  await dbUpdateChannel(updated);
  if (memoryChannels) {
    memoryChannels = memoryChannels.map((c) => (c.id === channelId ? updated : c));
  }
  if (memoryChannels && memoryRadio && memoryPrefs) {
    await syncFavoritesToRepo(memoryChannels, memoryRadio, memoryPrefs);
  }
  notifyLiveMediaSummary();
  return nextFav;
}

export async function toggleRadioFavorite(stationId: string): Promise<boolean> {
  let stations = memoryRadio ?? (await dbGetAllRadio()).map(enrichRadio);
  const st = stations.find((r) => r.id === stationId);
  if (!st) return false;
  const nextFav = !st.favorite;
  memoryPrefs = await setRadioFavorite(stationId, nextFav);
  const updated: RadioStation = { ...st, favorite: nextFav };
  await dbUpdateRadio(updated);
  if (memoryRadio) {
    memoryRadio = memoryRadio.map((r) => (r.id === stationId ? updated : r));
  }
  if (memoryChannels && memoryRadio && memoryPrefs) {
    await syncFavoritesToRepo(memoryChannels, memoryRadio, memoryPrefs);
  }
  notifyLiveMediaSummary();
  return nextFav;
}

export async function hideChannelFromCatalog(channelId: string): Promise<void> {
  memoryPrefs = await blacklistChannel(channelId);
  if (memoryChannels) {
    memoryChannels = memoryChannels.filter((c) => c.id !== channelId);
  }
  notifyLiveMediaSummary();
}

export async function hideRadioFromCatalog(stationId: string): Promise<void> {
  memoryPrefs = await blacklistRadio(stationId);
  if (memoryRadio) {
    memoryRadio = memoryRadio.filter((r) => r.id !== stationId);
  }
  notifyLiveMediaSummary();
}

export async function getLiveMediaUserPrefs(): Promise<LiveMediaUserPrefs> {
  if (memoryPrefs) return memoryPrefs;
  memoryPrefs = await loadUserPrefs();
  return memoryPrefs;
}

export async function exportLiveMediaUserPrefs(): Promise<string> {
  const prefs = await getLiveMediaUserPrefs();
  return exportUserPrefsJson(prefs);
}

export async function importLiveMediaUserPrefs(raw: string): Promise<void> {
  const prefs = importUserPrefsJson(raw);
  await saveUserPrefs(prefs);
  memoryPrefs = prefs;
  invalidateLiveMediaMemoryCache();
  await ensureLiveMediaLibrary();
  if (memoryChannels && memoryRadio && memoryPrefs) {
    await syncFavoritesToRepo(memoryChannels, memoryRadio, memoryPrefs);
  }
}

export async function restoreHiddenChannel(channelId: string): Promise<void> {
  memoryPrefs = await restoreChannelFromBlacklist(channelId);
  invalidateLiveMediaMemoryCache();
  await syncAllLiveMediaSources();
  await ensureLiveMediaLibrary();
}

export async function restoreHiddenRadio(stationId: string): Promise<void> {
  memoryPrefs = await restoreRadioFromBlacklist(stationId);
  invalidateLiveMediaMemoryCache();
  await syncAllLiveMediaSources();
  await ensureLiveMediaLibrary();
}

export async function saveChannelUserOverride(
  channelId: string,
  patch: ChannelUserOverride | null,
): Promise<void> {
  memoryPrefs = await setChannelOverride(channelId, patch);
  invalidateLiveMediaMemoryCache();
  await ensureLiveMediaLibrary();
}

export async function saveTunerPreferences(
  patch: Parameters<typeof updateTunerPreferences>[0],
): Promise<void> {
  memoryPrefs = await updateTunerPreferences(patch);
  invalidateLiveMediaMemoryCache();
}

export async function getFavoriteChannels(): Promise<Channel[]> {
  const channels = memoryChannels ?? (await dbGetAllChannels());
  return channels.filter((c) => c.favorite && (c.type === "tv" || c.type === "youtube"));
}

export async function getFavoriteRadio(): Promise<RadioStation[]> {
  const stations = memoryRadio ?? (await dbGetAllRadio());
  return stations.filter((r) => r.favorite);
}

export function invalidateLiveMediaMemoryCache(): void {
  memoryChannels = null;
  memoryRadio = null;
  memoryPrefs = null;
}

export function isLiveMediaBusy(): boolean {
  return syncPromise != null || validatePromise != null;
}

const FLAGS: Record<string, string> = {
  il: "🇮🇱", us: "🇺🇸", gb: "🇬🇧", de: "🇩🇪", fr: "🇫🇷", ru: "🇷🇺", jp: "🇯🇵",
};

export function countryFlag(code?: string): string {
  if (!code) return "";
  return FLAGS[code.toLowerCase()] || "🌍";
}
