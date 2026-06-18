import { parseYouTubeVideoId } from "../youtubeUrls";
import type { MediaSerpHit } from "../types";
import { searchInvidiousVideos } from "./invidiousMedia";
import { searchPipedVideos } from "./pipedMedia";
import { searchSearxngYouTubeMedia } from "./searxngYoutube";

export const dedupeYouTubeMediaHits = (hits: MediaSerpHit[]): MediaSerpHit[] => {
  const seen = new Set<string>();
  const out: MediaSerpHit[] = [];
  for (const hit of hits) {
    const key = parseYouTubeVideoId(hit.url) ?? hit.id;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(hit);
  }
  return out;
};

/** Invidious + Piped + SearXNG site:youtube — maximum YouTube coverage without API key. */
export async function searchAllYouTubeMedia(query: string): Promise<MediaSerpHit[]> {
  const [invidious, piped, searx] = await Promise.all([
    searchInvidiousVideos(query).catch(() => [] as MediaSerpHit[]),
    searchPipedVideos(query).catch(() => [] as MediaSerpHit[]),
    searchSearxngYouTubeMedia(query).catch(() => [] as MediaSerpHit[]),
  ]);
  return dedupeYouTubeMediaHits([...invidious, ...piped, ...searx]);
}
