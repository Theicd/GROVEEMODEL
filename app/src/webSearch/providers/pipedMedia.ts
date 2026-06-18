import { buildYouTubeSearchQuery } from "../intents";
import { fetchJson } from "../fetchJson";
import { parseYouTubeVideoId, youtubeEmbedUrl, youtubeThumbnail, youtubeWatchUrl } from "../youtubeUrls";
import type { MediaSerpHit } from "../types";

type PipedSearchItem = {
  type?: string;
  title?: string;
  url?: string;
  thumbnail?: string;
  uploaderName?: string;
  uploaderUrl?: string;
  duration?: number;
  playlistId?: string;
  videos?: number;
};

type PipedSearchResponse = { items?: PipedSearchItem[] };

const DEFAULT_PIPED = [
  "https://pipedapi.adminforge.de",
  "https://api.piped.projectsegfau.lt",
  "https://pipedapi.in.projectsegfau.lt",
  "https://pipedapi.kavin.rocks",
];

const getPipedInstances = (): string[] => {
  const env = (import.meta.env.VITE_PIPED_INSTANCES as string | undefined)?.trim();
  const fromEnv = env
    ? env
        .split(",")
        .map((s) => s.trim().replace(/\/$/, ""))
        .filter(Boolean)
    : [];
  return [...new Set([...fromEnv, ...DEFAULT_PIPED])];
};

const videoIdFromPipedUrl = (raw?: string): string | null => {
  if (!raw?.trim()) return null;
  const url = raw.startsWith("http") ? raw : `https://www.youtube.com${raw.startsWith("/") ? "" : "/"}${raw}`;
  return parseYouTubeVideoId(url);
};

export function mapPipedStream(item: PipedSearchItem): MediaSerpHit | null {
  if (item.type !== "stream" || !item.title?.trim()) return null;
  const videoId = videoIdFromPipedUrl(item.url);
  if (!videoId) return null;
  return {
    id: `piped-${videoId}`,
    mediaType: "video",
    youtubeSubType: "video",
    title: item.title.trim(),
    url: youtubeWatchUrl(videoId),
    playUrl: youtubeEmbedUrl(videoId),
    thumbnail: item.thumbnail?.trim() || youtubeThumbnail(videoId),
    snippet: item.uploaderName ? `ערוץ: ${item.uploaderName}` : "",
    author: item.uploaderName,
    source: "YouTube",
    durationSec: item.duration,
  };
}

export function mapPipedPlaylist(item: PipedSearchItem): MediaSerpHit | null {
  if (item.type !== "playlist" || !item.title?.trim()) return null;
  const playlistId =
    item.playlistId?.trim() || item.url?.match(/[?&]list=([^&]+)/)?.[1]?.trim();
  if (!playlistId) return null;
  return {
    id: `piped-pl-${playlistId}`,
    mediaType: "video",
    youtubeSubType: "playlist",
    title: item.title.trim(),
    url: `https://www.youtube.com/playlist?list=${playlistId}`,
    playUrl: "",
    thumbnail: item.thumbnail?.trim() || "",
    snippet: item.videos ? `פלייליסט · ${item.videos} סרטונים` : "פלייליסט",
    author: item.uploaderName,
    source: "YouTube",
  };
};

export function mapPipedChannel(item: PipedSearchItem): MediaSerpHit | null {
  if (item.type !== "channel" || !item.uploaderName?.trim()) return null;
  const channelUrl = item.uploaderUrl?.trim() || item.url?.trim();
  if (!channelUrl) return null;
  const fullUrl = channelUrl.startsWith("http")
    ? channelUrl
    : `https://www.youtube.com${channelUrl.startsWith("/") ? "" : "/"}${channelUrl}`;
  return {
    id: `piped-ch-${item.uploaderName.replace(/\s+/g, "-")}`,
    mediaType: "video",
    youtubeSubType: "channel",
    title: item.uploaderName.trim(),
    url: fullUrl,
    playUrl: "",
    thumbnail: item.thumbnail?.trim() || "",
    snippet: "ערוץ YouTube",
    author: item.uploaderName,
    source: "YouTube",
  };
};

const mapPipedItem = (item: PipedSearchItem): MediaSerpHit | null => {
  if (item.type === "playlist") return mapPipedPlaylist(item);
  if (item.type === "channel") return mapPipedChannel(item);
  return mapPipedStream(item);
};

export async function searchPipedVideos(query: string): Promise<MediaSerpHit[]> {
  const q = buildYouTubeSearchQuery(query) || query.trim();
  if (!q) return [];

  for (const base of getPipedInstances()) {
    try {
      const url = `${base}/search?q=${encodeURIComponent(q)}&filter=all`;
      const data = await fetchJson<PipedSearchResponse>(url, undefined, { timeoutMs: 10_000 });
      const hits = (data.items ?? [])
        .slice(0, 12)
        .map(mapPipedItem)
        .filter((h): h is MediaSerpHit => h != null);
      if (hits.length) return hits;
    } catch {
      continue;
    }
  }
  return [];
};
