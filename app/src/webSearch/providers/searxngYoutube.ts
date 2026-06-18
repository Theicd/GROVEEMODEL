import { buildYouTubeSearchQuery } from "../intents";
import { fetchJson } from "../fetchJson";
import { parseYouTubeVideoId, youtubeEmbedUrl, youtubeThumbnail, youtubeWatchUrl } from "../youtubeUrls";
import type { MediaSerpHit, WebSerpHit } from "../types";
import { getSearxngBaseUrl } from "./searxng";

type SearxResult = {
  results?: Array<{ title?: string; url?: string; content?: string; engine?: string }>;
};

const toWebHits = (
  rows: Array<{ title?: string; url?: string; content?: string; engine?: string }>,
): WebSerpHit[] =>
  rows
    .filter((r) => r.url?.trim())
    .map((r, i) => ({
      id: `searxng-yt-${i}-${(r.url ?? "").slice(0, 48)}`,
      title: (r.title ?? "ללא כותרת").trim(),
      url: r.url!.trim(),
      snippet: (r.content ?? "").replace(/\s+/g, " ").trim().slice(0, 280),
      engine: r.engine,
    }));

export function webHitToYouTubeMedia(hit: WebSerpHit): MediaSerpHit | null {
  const videoId = parseYouTubeVideoId(hit.url);
  if (!videoId) return null;
  return {
    id: `searx-yt-${videoId}`,
    mediaType: "video",
    youtubeSubType: "video",
    title: hit.title,
    url: youtubeWatchUrl(videoId),
    playUrl: youtubeEmbedUrl(videoId),
    thumbnail: youtubeThumbnail(videoId),
    snippet: hit.snippet,
    source: "YouTube",
  };
}

async function searxngQuery(q: string, categories: string): Promise<WebSerpHit[]> {
  const base = getSearxngBaseUrl();
  if (!base) return [];
  const params = new URLSearchParams({
    q,
    format: "json",
    language: "he-IL",
    categories,
  });
  const url = `${base}/search?${params.toString()}`;
  const data = await fetchJson<SearxResult>(url, undefined, { timeoutMs: 12_000 });
  return toWebHits((data.results ?? []).slice(0, 15));
}

/** YouTube-focused SearXNG queries — works when Invidious/Piped mirrors are down. */
export async function searchSearxngYouTubeMedia(query: string): Promise<MediaSerpHit[]> {
  const terms = buildYouTubeSearchQuery(query) || query.trim();
  if (!terms) return [];

  const searches: { q: string; categories: string }[] = [
    { q: `${terms} site:youtube.com`, categories: "general" },
    { q: terms, categories: "videos" },
    { q: `${terms} שיר`, categories: "videos" },
  ];

  const seen = new Set<string>();
  const out: MediaSerpHit[] = [];

  for (const { q, categories } of searches) {
    try {
      const webHits = await searxngQuery(q, categories);
      for (const w of webHits) {
        const media = webHitToYouTubeMedia(w);
        if (!media) continue;
        const key = parseYouTubeVideoId(media.url) ?? media.id;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(media);
      }
    } catch {
      continue;
    }
    if (out.length >= 15) break;
  }

  return out.slice(0, 15);
}
