import type { SearchProviderId } from "../webSearch/types";
import {
  isYouTubeUrl,
  parseYouTubeVideoId,
  youtubeEmbedUrl,
  youtubeThumbnail,
  youtubeWatchUrl,
} from "../webSearch/youtubeUrls";
import { cleanDisplaySnippet } from "./snippetCleanup";
import type { UnifiedSearchHit } from "./types";

const YT_FAVICON = "https://www.youtube.com/favicon.ico";

export const isYoutubeHit = (hit: UnifiedSearchHit): boolean =>
  hit.kind === "youtube" ||
  hit.provider === "invidious-videos" ||
  isYouTubeUrl(hit.url);

export function youtubeHitFromWeb(w: {
  id: string;
  title: string;
  url: string;
  snippet: string;
  score?: number;
}): UnifiedSearchHit | null {
  if (!isYouTubeUrl(w.url)) return null;
  const videoId = parseYouTubeVideoId(w.url);
  const title = w.title.trim();
  const snippet = cleanDisplaySnippet(title, w.snippet, w.url);
  if (videoId) {
    return {
      id: `yt-web-${videoId}`,
      kind: "youtube",
      title,
      url: youtubeWatchUrl(videoId),
      snippet,
      imageUrl: youtubeThumbnail(videoId),
      mediaPlayUrl: youtubeEmbedUrl(videoId),
      mediaEmbedMode: true,
      sourceLabel: "YouTube",
      faviconUrl: YT_FAVICON,
      provider: "searxng",
      score: w.score ?? 52,
      summarizable: false,
    };
  }
  return {
    id: w.id,
    kind: "youtube",
    title,
    url: w.url,
    snippet,
    sourceLabel: "YouTube",
    faviconUrl: YT_FAVICON,
    provider: "searxng",
    score: w.score ?? 48,
    summarizable: false,
  };
}

export function youtubeHitFromMedia(m: {
  id: string;
  title: string;
  url: string;
  snippet?: string;
  thumbnail?: string;
  playUrl?: string;
  author?: string;
  durationSec?: number;
  youtubeSubType?: "video" | "playlist" | "channel";
  score: number;
  provider: SearchProviderId;
}): UnifiedSearchHit {
  const playable = Boolean(m.playUrl?.trim()) && m.youtubeSubType !== "playlist" && m.youtubeSubType !== "channel";
  const videoId = parseYouTubeVideoId(m.url);
  return {
    id: m.id,
    kind: "youtube",
    title: m.title,
    titleOriginal: m.title,
    url: videoId ? youtubeWatchUrl(videoId) : m.url,
    snippet: m.snippet?.trim() || "",
    snippetOriginal: m.snippet || "",
    imageUrl: m.thumbnail || (videoId ? youtubeThumbnail(videoId) : undefined),
    mediaPlayUrl: playable ? m.playUrl : undefined,
    mediaEmbedMode: playable,
    durationSec: m.durationSec,
    author: m.author,
    sourceLabel: "YouTube",
    faviconUrl: YT_FAVICON,
    provider: m.provider,
    score: m.score,
    meta: m.youtubeSubType && m.youtubeSubType !== "video" ? { engine: m.youtubeSubType } : undefined,
    summarizable: false,
  };
}
