import type { Channel, RadioStation } from "./types";
import type { LiveMediaUserPrefs } from "./userPrefs";

export const CURATED_FAVORITES_PUBLIC_PATH = "liveMedia/curatedFavorites.json";
export const CURATED_FAVORITES_API_PATH = "/api/live-media/curated-favorites";

export type CuratedFavoriteChannel = {
  id: string;
  name: string;
  country: string;
  language: string;
  category: string;
  stream: string;
  source: string;
  type: Channel["type"];
  tags?: string[];
};

export type CuratedFavoriteRadio = {
  id: string;
  name: string;
  country: string;
  countrycode: string;
  language: string;
  stream: string;
  tags: string[];
};

export type CuratedFavoritesFile = {
  version: 1;
  description?: string;
  updatedAt: number;
  channels: CuratedFavoriteChannel[];
  radio: CuratedFavoriteRadio[];
};

export function emptyCuratedFavoritesFile(): CuratedFavoritesFile {
  return {
    version: 1,
    description:
      "Curated TV/radio favorites — source of truth in git. Auto-updated when starring in dev (npm run dev).",
    updatedAt: 0,
    channels: [],
    radio: [],
  };
}

function curatedFavoritesUrl(): string {
  const base = import.meta.env.BASE_URL || "./";
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}${CURATED_FAVORITES_PUBLIC_PATH}`;
}

export async function fetchCuratedFavoritesFromRepo(): Promise<CuratedFavoritesFile | null> {
  try {
    const res = await fetch(curatedFavoritesUrl(), { cache: "no-store" });
    if (!res.ok) return null;
    const parsed = (await res.json()) as CuratedFavoritesFile;
    if (parsed.version !== 1 || !Array.isArray(parsed.channels) || !Array.isArray(parsed.radio)) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function channelToCuratedSnapshot(channel: Channel): CuratedFavoriteChannel {
  return {
    id: channel.id,
    name: channel.name,
    country: channel.country,
    language: channel.language,
    category: channel.category,
    stream: channel.stream,
    source: channel.source,
    type: channel.type,
    tags: channel.tags,
  };
}

export function radioToCuratedSnapshot(station: RadioStation): CuratedFavoriteRadio {
  return {
    id: station.id,
    name: station.name,
    country: station.country,
    countrycode: station.countrycode,
    language: station.language,
    stream: station.stream,
    tags: station.tags ?? [],
  };
}

export function buildCuratedFavoritesFile(
  channels: Channel[],
  radio: RadioStation[],
  prefs: LiveMediaUserPrefs,
): CuratedFavoritesFile {
  const favTv = new Set(prefs.favoriteChannelIds);
  const favRadio = new Set(prefs.favoriteRadioIds);
  const channelById = new Map(channels.map((c) => [c.id, c]));
  const radioById = new Map(radio.map((r) => [r.id, r]));

  const curatedChannels = prefs.favoriteChannelIds
    .map((id) => channelById.get(id))
    .filter((c): c is Channel => Boolean(c))
    .map(channelToCuratedSnapshot)
    .sort((a, b) => a.name.localeCompare(b.name, "he"));

  const curatedRadio = prefs.favoriteRadioIds
    .map((id) => radioById.get(id))
    .filter((r): r is RadioStation => Boolean(r))
    .map(radioToCuratedSnapshot)
    .sort((a, b) => a.name.localeCompare(b.name, "he"));

  return {
    version: 1,
    description:
      "Curated TV/radio favorites — source of truth in git. Auto-updated when starring in dev (npm run dev).",
    updatedAt: Date.now(),
    channels: curatedChannels,
    radio: curatedRadio,
  };
}

export function mergeCuratedFavoritesIntoPrefs(
  prefs: LiveMediaUserPrefs,
  curated: CuratedFavoritesFile | null,
): { prefs: LiveMediaUserPrefs; changed: boolean } {
  if (!curated) return { prefs, changed: false };

  const favoriteChannelIds = [
    ...new Set([...prefs.favoriteChannelIds, ...curated.channels.map((c) => c.id)]),
  ];
  const favoriteRadioIds = [
    ...new Set([...prefs.favoriteRadioIds, ...curated.radio.map((r) => r.id)]),
  ];

  const changed =
    favoriteChannelIds.length !== prefs.favoriteChannelIds.length ||
    favoriteRadioIds.length !== prefs.favoriteRadioIds.length;

  if (!changed) return { prefs, changed: false };

  return {
    prefs: {
      ...prefs,
      favoriteChannelIds,
      favoriteRadioIds,
      updatedAt: Date.now(),
    },
    changed: true,
  };
}

export async function persistCuratedFavoritesToRepo(
  channels: Channel[],
  radio: RadioStation[],
  prefs: LiveMediaUserPrefs,
): Promise<{ ok: boolean; skipped?: boolean; error?: string }> {
  if (!import.meta.env.DEV) return { ok: true, skipped: true };

  const body = buildCuratedFavoritesFile(channels, radio, prefs);
  try {
    const res = await fetch(CURATED_FAVORITES_API_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      return { ok: false, error: text || `HTTP ${res.status}` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
