import { buildMediaSearchQuery, buildMoviesSearchQuery } from "../intents";
import { fetchJson } from "../fetchJson";
import type { MediaSerpHit, SearchSourceResult } from "../types";

type PeerTubeVideo = {
  uuid?: string;
  name?: string;
  description?: string;
  truncatedDescription?: string;
  duration?: number;
  url?: string;
  thumbnailUrl?: string;
  thumbnailPath?: string;
  account?: { displayName?: string; name?: string };
  channel?: { displayName?: string; name?: string };
};

type PeerTubeSearchResponse = {
  total?: number;
  data?: PeerTubeVideo[];
};

type PeerTubeFile = {
  fileUrl?: string;
  resolution?: { label?: string };
  width?: number;
};

type PeerTubeVideoDetail = {
  streamingPlaylists?: Array<{ files?: PeerTubeFile[] }>;
};

const DEFAULT_SEPIA = "https://sepiasearch.org";
const MAX_RESULTS = 12;
const MAX_ENRICH = 8;

const getSepiaBase = (): string => {
  const env = (import.meta.env.VITE_SEPIA_SEARCH_URL as string | undefined)?.trim();
  return (env || DEFAULT_SEPIA).replace(/\/$/, "");
};

export const peerTubeSearchQuery = (query: string): string => {
  const cleaned = buildMediaSearchQuery(query) || buildMoviesSearchQuery(query) || query.trim();
  return cleaned.length >= 2 ? cleaned : "";
};

export function peerTubeInstanceOrigin(videoUrl: string): string | null {
  try {
    return new URL(videoUrl).origin;
  } catch {
    return null;
  }
}

export function pickPeerTubePlayUrl(files: PeerTubeFile[]): string | null {
  const mp4 = files.filter((f) => f.fileUrl?.includes(".mp4"));
  if (!mp4.length) return files[0]?.fileUrl?.trim() || null;

  const score = (f: PeerTubeFile): number => {
    const label = f.resolution?.label ?? "";
    let s = 0;
    if (label === "720p") s += 30;
    else if (label === "480p") s += 28;
    else if (label === "1080p") s += 22;
    else if (label === "360p") s += 20;
    if ((f.width ?? 0) <= 1280) s += 5;
    return s;
  };

  return [...mp4].sort((a, b) => score(b) - score(a))[0]?.fileUrl?.trim() || null;
}

export function mapPeerTubeSearchHit(video: PeerTubeVideo, playUrl?: string): MediaSerpHit | null {
  const uuid = video.uuid?.trim();
  const title = video.name?.trim();
  const pageUrl = video.url?.trim();
  if (!uuid || !title || !pageUrl) return null;

  const thumb =
    video.thumbnailUrl?.trim() ||
    (video.thumbnailPath && pageUrl
      ? `${peerTubeInstanceOrigin(pageUrl) ?? ""}${video.thumbnailPath}`
      : "");

  const author = video.account?.displayName || video.account?.name || video.channel?.displayName;

  return {
    id: `peertube-${uuid}`,
    mediaType: "video",
    title,
    url: pageUrl,
    playUrl: playUrl || pageUrl,
    thumbnail: thumb,
    snippet: (video.truncatedDescription || video.description || "").slice(0, 280),
    author,
    source: "PeerTube",
    durationSec: video.duration,
  };
}

export async function resolvePeerTubePlayUrl(
  videoUrl: string,
  uuid: string,
): Promise<string | null> {
  const origin = peerTubeInstanceOrigin(videoUrl);
  if (!origin) return null;
  try {
    const detail = await fetchJson<PeerTubeVideoDetail>(
      `${origin}/api/v1/videos/${encodeURIComponent(uuid)}`,
      undefined,
      { timeoutMs: 6000 },
    );
    const files = detail.streamingPlaylists?.[0]?.files ?? [];
    return pickPeerTubePlayUrl(files);
  } catch {
    return null;
  }
}

export async function searchPeerTubeVideos(query: string): Promise<MediaSerpHit[]> {
  const q = peerTubeSearchQuery(query);
  if (!q) return [];

  const base = getSepiaBase();
  const data = await fetchJson<PeerTubeSearchResponse>(
    `${base}/api/v1/search/videos?search=${encodeURIComponent(q)}&count=${MAX_RESULTS}`,
    undefined,
    { timeoutMs: 12_000 },
  );

  const rows = data.data ?? [];
  if (!rows.length) return [];

  const enrichRows = rows.slice(0, MAX_ENRICH);
  const playByUuid = new Map<string, string>();

  await Promise.all(
    enrichRows.map(async (video) => {
      const uuid = video.uuid?.trim();
      const url = video.url?.trim();
      if (!uuid || !url) return;
      const play = await resolvePeerTubePlayUrl(url, uuid);
      if (play) playByUuid.set(uuid, play);
    }),
  );

  return rows
    .map((video) => {
      const uuid = video.uuid?.trim();
      const play = uuid ? playByUuid.get(uuid) : undefined;
      return mapPeerTubeSearchHit(video, play);
    })
    .filter((h): h is MediaSerpHit => h != null);
}

const emptyResult = (error: string, started: number): SearchSourceResult => ({
  provider: "peertube-videos",
  label: "PeerTube · וידאו",
  ok: false,
  text: "",
  error,
  latencyMs: Math.round(performance.now() - started),
});

export const fetchPeerTubeVideosSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "peertube-videos" as const;
  const label = "PeerTube · וידאו";

  const q = peerTubeSearchQuery(query);
  if (!q) {
    return emptyResult("אין שאילתת וידאו מתאימה", started);
  }

  try {
    const hits = await searchPeerTubeVideos(query);
    if (!hits.length) {
      return emptyResult(`לא נמצאו סרטוני PeerTube עבור: ${q}`, started);
    }

    const lines = [`שאילתה: ${q} · PeerTube (Sepia Search)`, ...hits.map((h, i) => `${i + 1}. ${h.title}`)];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: hits[0]?.url,
      mediaHits: hits,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return emptyResult(err instanceof Error ? err.message : "שגיאה בחיפוש PeerTube", started);
  }
};
