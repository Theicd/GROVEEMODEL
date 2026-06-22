import type { MediaSerpHit, WebSerpHit } from "../types";
import { fetchArchiveVideoPlayback, parseArchiveIdentifier, promoteArchiveWebHitsToMedia } from "./archiveOrgVideo";

const MAX_ASYNC_VIDEO_LOOKUPS = 4;

const VIMEO_RE = /vimeo\.com\/(?:video\/)?(\d+)/i;
const DAILY_RE = /dailymotion\.com\/video\/([a-z0-9]+)/i;
const PEERTUBE_WATCH_RE = /\/videos\/watch\/([a-f0-9-]{8,})/i;
const DIRECT_VIDEO_RE = /\.(mp4|webm|ogv|m4v)(\?|#|$)/i;

const vimeoEmbed = (id: string): string => `https://player.vimeo.com/video/${id}`;
const dailymotionEmbed = (id: string): string => `https://www.dailymotion.com/embed/video/${id}`;

const peertubeEmbedFromUrl = (url: string): string | null => {
  try {
    const u = new URL(url);
    const m = u.pathname.match(PEERTUBE_WATCH_RE);
    if (!m) return null;
    return `${u.origin}/videos/embed/${m[1]}`;
  } catch {
    return null;
  }
};

const syncVideoFromUrl = (hit: WebSerpHit): MediaSerpHit | null => {
  const url = hit.url.trim();
  if (!url) return null;

  const vimeo = url.match(VIMEO_RE);
  if (vimeo) {
    return {
      id: `openserp-vimeo-${vimeo[1]}`,
      mediaType: "video",
      title: hit.title,
      url,
      playUrl: vimeoEmbed(vimeo[1]),
      thumbnail: `https://vumbnail.com/${vimeo[1]}.jpg`,
      snippet: hit.snippet,
      source: "OpenSERP",
    };
  }

  const dm = url.match(DAILY_RE);
  if (dm) {
    return {
      id: `openserp-dm-${dm[1]}`,
      mediaType: "video",
      title: hit.title,
      url,
      playUrl: dailymotionEmbed(dm[1]),
      snippet: hit.snippet,
      source: "OpenSERP",
    };
  }

  const ptEmbed = peertubeEmbedFromUrl(url);
  if (ptEmbed) {
    return {
      id: `openserp-pt-${hit.id}`,
      mediaType: "video",
      title: hit.title,
      url,
      playUrl: ptEmbed,
      snippet: hit.snippet,
      source: "PeerTube",
    };
  }

  if (DIRECT_VIDEO_RE.test(url)) {
    return {
      id: `openserp-vid-${hit.id}`,
      mediaType: "video",
      title: hit.title,
      url,
      playUrl: url,
      snippet: hit.snippet,
      source: "OpenSERP",
    };
  }

  return null;
};

/** Promote playable video links found in companion web SERP (Archive, Vimeo, PeerTube, direct files). */
export const promoteCompanionWebHitsToMedia = async (
  hits: WebSerpHit[],
): Promise<{ webHits: WebSerpHit[]; mediaHits: MediaSerpHit[] }> => {
  const { webHits: afterArchive, mediaHits: archiveMedia } = await promoteArchiveWebHitsToMedia(hits);

  const webHits: WebSerpHit[] = [];
  const mediaHits: MediaSerpHit[] = [...archiveMedia];
  const archiveUrls = new Set(archiveMedia.map((m) => m.url.toLowerCase()));

  for (const hit of afterArchive) {
    const sync = syncVideoFromUrl(hit);
    if (sync && !archiveUrls.has(hit.url.toLowerCase())) {
      mediaHits.push(sync);
      continue;
    }
    webHits.push(hit);
  }

  return { webHits, mediaHits };
};

export { parseArchiveIdentifier, fetchArchiveVideoPlayback };
