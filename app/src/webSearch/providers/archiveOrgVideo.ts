import { fetchJson } from "../fetchJson";
import type { MediaSerpHit, MovieSerpHit, WebSerpHit } from "../types";

type ArchiveFile = {
  name?: string;
  format?: string;
  size?: string | number;
};

type ArchiveMetadata = {
  metadata?: { identifier?: string; mediatype?: string; runtime?: string };
  files?: ArchiveFile[];
};

const VIDEO_FORMAT_RE = /mpeg4|h\.?264|webm|ogv|video|matroska|quicktime/i;
const VIDEO_EXT_RE = /\.(mp4|webm|ogv|mov|m4v)$/i;

/** Parse `archive.org/details/{identifier}` from a URL. */
export function parseArchiveIdentifier(url: string): string | null {
  const m = url.match(/archive\.org\/details\/([^/?#]+)/i);
  return m ? decodeURIComponent(m[1]) : null;
}

export function archiveThumbnailUrl(identifier: string): string {
  return `https://archive.org/services/img/${identifier}`;
}

export function archiveDownloadUrl(identifier: string, filename: string): string {
  return `https://archive.org/download/${encodeURIComponent(identifier)}/${encodeURIComponent(filename)}`;
}

/** Pick a browser-playable file from IA metadata (prefer mp4). */
export function pickBestVideoFile(files: ArchiveFile[]): ArchiveFile | null {
  const candidates = files.filter((f) => {
    const name = f.name ?? "";
    const format = f.format ?? "";
    if (/\.(xml|json|torrent|jpg|jpeg|png|gif|srt|vtt|log)$/i.test(name)) return false;
    return VIDEO_FORMAT_RE.test(format) || VIDEO_EXT_RE.test(name);
  });
  if (!candidates.length) return null;

  const scored = candidates.map((f) => {
    const name = f.name ?? "";
    let score = 0;
    if (/\.mp4$/i.test(name)) score += 20;
    if (!/\.ia\.mp4$/i.test(name)) score += 8;
    if (/MPEG4/i.test(f.format ?? "")) score += 5;
    const size = Number(f.size) || 0;
    if (size > 0 && size < 80_000_000) score += 2;
    return { f, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored[0]?.f ?? null;
}

function parseRuntimeSeconds(runtime?: string): number | undefined {
  if (!runtime?.trim()) return undefined;
  const parts = runtime.trim().split(":").map((p) => Number(p));
  if (parts.some((n) => !Number.isFinite(n))) return undefined;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 1) return parts[0];
  return undefined;
}

export async function fetchArchiveVideoPlayback(
  identifier: string,
): Promise<{ playUrl: string; thumbnail: string; durationSec?: number } | null> {
  try {
    const data = await fetchJson<ArchiveMetadata>(
      `https://archive.org/metadata/${encodeURIComponent(identifier)}`,
      undefined,
      { timeoutMs: 8000 },
    );
    const file = pickBestVideoFile(data.files ?? []);
    if (!file?.name) return null;
    return {
      playUrl: archiveDownloadUrl(identifier, file.name),
      thumbnail: archiveThumbnailUrl(identifier),
      durationSec: parseRuntimeSeconds(data.metadata?.runtime),
    };
  } catch {
    return null;
  }
}

/** Attach direct mp4 URL to Internet Archive movie rows when available. */
export async function enrichMovieHitsWithArchiveVideo(hits: MovieSerpHit[]): Promise<MovieSerpHit[]> {
  return Promise.all(
    hits.map(async (hit) => {
      const fromUrl = parseArchiveIdentifier(hit.url);
      const fromId = hit.id.startsWith("archive-") ? hit.id.slice("archive-".length) : null;
      const identifier = fromUrl || fromId;
      if (!identifier) return hit;
      if (hit.source && hit.source !== "Internet Archive") return hit;

      const playback = await fetchArchiveVideoPlayback(identifier);
      if (!playback) return hit;

      return {
        ...hit,
        playUrl: playback.playUrl,
        durationSec: playback.durationSec,
        poster: hit.poster || playback.thumbnail,
      };
    }),
  );
}

const MAX_ARCHIVE_WEB_PROMOTIONS = 4;

/** Promote SearXNG web rows that point at IA details pages with playable video. */
export async function promoteArchiveWebHitsToMedia(
  hits: WebSerpHit[],
): Promise<{ webHits: WebSerpHit[]; mediaHits: MediaSerpHit[] }> {
  const webHits: WebSerpHit[] = [];
  const mediaHits: MediaSerpHit[] = [];
  const candidates = hits
    .map((h) => ({ hit: h, id: parseArchiveIdentifier(h.url) }))
    .filter((x): x is { hit: WebSerpHit; id: string } => !!x.id);

  const uniqueIds = [...new Set(candidates.map((c) => c.id))].slice(0, MAX_ARCHIVE_WEB_PROMOTIONS);
  const playbackById = new Map<string, Awaited<ReturnType<typeof fetchArchiveVideoPlayback>>>();

  await Promise.all(
    uniqueIds.map(async (id) => {
      const playback = await fetchArchiveVideoPlayback(id);
      if (playback) playbackById.set(id, playback);
    }),
  );

  for (const h of hits) {
    const id = parseArchiveIdentifier(h.url);
    const playback = id ? playbackById.get(id) : undefined;
    if (id && playback) {
      mediaHits.push({
        id: `archive-web-vid-${id}`,
        mediaType: "video",
        title: h.title,
        url: h.url,
        playUrl: playback.playUrl,
        thumbnail: playback.thumbnail,
        snippet: h.snippet,
        source: "Internet Archive",
        durationSec: playback.durationSec,
      });
    } else {
      webHits.push(h);
    }
  }

  return { webHits, mediaHits };
}
