import { fetchJson } from "../fetchJson";
import { buildMediaSearchQuery, buildMoviesSearchQuery } from "../intents";
import type { MediaSerpHit, MovieSerpHit } from "../types";
import {
  archiveThumbnailUrl,
  enrichMovieHitsWithArchiveVideo,
  fetchArchiveVideoPlayback,
} from "./archiveOrgVideo";

type ArchiveDoc = {
  identifier?: string;
  title?: string;
  description?: string;
  year?: number;
  mediatype?: string;
};

type ArchiveSearchResponse = {
  response?: { docs?: ArchiveDoc[] };
};

const DEFAULT_ROWS = 12;
const MAX_ENRICH_PLAYBACK = 8;

const escapeArchiveTerm = (term: string): string => term.replace(/"/g, '\\"').trim();

/** Build IA advancedsearch clauses — Hebrew, TV archive, and general video. */
export function buildArchiveSearchQueries(rawQuery: string, cleaned: string): string[] {
  const term = escapeArchiveTerm(cleaned);
  if (!term) return [];

  const textClause = `(title:${term} OR description:${term})`;
  const queries = [
    `(mediatype:movies OR mediatype:video OR mediatype:tv) AND ${textClause}`,
  ];

  if (/[\u0590-\u05FF]/.test(cleaned)) {
    queries.push(`(mediatype:movies OR mediatype:video) AND language:Hebrew AND ${textClause}`);
  }

  if (/ערוץ\s*11|כאן|channel\s*11|ערוץ\s*החינוך|טלוויזיה|תוכנית/i.test(rawQuery)) {
    queries.push(
      `(mediatype:movies OR mediatype:video) AND (title:ערוץ OR title:"channel 11" OR subject:כאן OR subject:ישראל OR subject:Israel)`,
    );
  }

  return [...new Set(queries)];
}

async function fetchArchiveDocs(query: string, rows: number): Promise<ArchiveDoc[]> {
  const data = await fetchJson<ArchiveSearchResponse>(
    `https://archive.org/advancedsearch.php?q=${encodeURIComponent(query)}` +
      `&fl[]=identifier,title,description,year,mediatype&rows=${rows}&output=json`,
    undefined,
    { timeoutMs: 12_000 },
  );
  return (data.response?.docs ?? []).filter((d) => d.identifier && d.title);
}

const dedupeDocs = (docs: ArchiveDoc[]): ArchiveDoc[] => {
  const seen = new Set<string>();
  const out: ArchiveDoc[] = [];
  for (const doc of docs) {
    const id = doc.identifier!;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(doc);
  }
  return out;
};

export async function searchArchiveVideoDocs(
  rawQuery: string,
  options: { rowsPerQuery?: number; maxResults?: number } = {},
): Promise<ArchiveDoc[]> {
  const cleaned =
    buildMediaSearchQuery(rawQuery) || buildMoviesSearchQuery(rawQuery) || rawQuery.trim();
  if (!cleaned || cleaned.length < 2) return [];

  const rowsPerQuery = options.rowsPerQuery ?? DEFAULT_ROWS;
  const maxResults = options.maxResults ?? 20;
  const queries = buildArchiveSearchQueries(rawQuery, cleaned);
  if (!queries.length) return [];

  const batches = await Promise.all(
    queries.map((q) => fetchArchiveDocs(q, rowsPerQuery).catch(() => [] as ArchiveDoc[])),
  );
  return dedupeDocs(batches.flat()).slice(0, maxResults);
}

const docToMovieHit = (doc: ArchiveDoc): MovieSerpHit => ({
  id: `archive-${doc.identifier}`,
  title: String(doc.title),
  originalTitle: String(doc.title),
  year: typeof doc.year === "number" ? doc.year : undefined,
  url: `https://archive.org/details/${doc.identifier}`,
  snippet: (doc.description ? String(doc.description) : "ארכיון וידאו ב-Internet Archive").slice(
    0,
    480,
  ),
  poster: archiveThumbnailUrl(doc.identifier!),
  source: "Internet Archive",
});

const docToMediaHit = (doc: ArchiveDoc, playback?: { playUrl: string; durationSec?: number }): MediaSerpHit => ({
  id: `archive-vid-${doc.identifier}`,
  mediaType: "video",
  title: String(doc.title),
  url: `https://archive.org/details/${doc.identifier}`,
  playUrl: playback?.playUrl ?? `https://archive.org/details/${doc.identifier}`,
  thumbnail: archiveThumbnailUrl(doc.identifier!),
  snippet: (doc.description ? String(doc.description) : "").slice(0, 280),
  source: "Internet Archive",
  durationSec: playback?.durationSec,
});

/** Movie-catalog rows with optional in-browser playback URLs. */
export async function searchArchiveMovieHits(
  query: string,
  maxResults = 6,
): Promise<MovieSerpHit[]> {
  const docs = await searchArchiveVideoDocs(query, { rowsPerQuery: 8, maxResults });
  const rows = docs.map(docToMovieHit);
  return enrichMovieHitsWithArchiveVideo(rows);
}

/** Federated video search — returns playable media hits when metadata allows. */
export async function searchArchiveMediaHits(
  rawQuery: string,
  maxResults = 16,
): Promise<MediaSerpHit[]> {
  const docs = await searchArchiveVideoDocs(rawQuery, { rowsPerQuery: 12, maxResults });
  if (!docs.length) return [];

  const enrichTargets = docs.slice(0, MAX_ENRICH_PLAYBACK);
  const playbackById = new Map<string, Awaited<ReturnType<typeof fetchArchiveVideoPlayback>>>();

  await Promise.all(
    enrichTargets.map(async (doc) => {
      const id = doc.identifier!;
      const playback = await fetchArchiveVideoPlayback(id);
      if (playback) playbackById.set(id, playback);
    }),
  );

  return docs
    .map((doc) => {
      const playback = playbackById.get(doc.identifier!);
      if (!playback?.playUrl) return null;
      return docToMediaHit(doc, playback);
    })
    .filter((h): h is MediaSerpHit => h != null);
}
