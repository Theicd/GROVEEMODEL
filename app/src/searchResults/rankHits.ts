import type { UnifiedSearchHit } from "./types";
import { isMusicQuery, shouldSearchYouTube } from "../webSearch/intents";

const normalizeUrl = (url: string): string => {
  try {
    const u = new URL(url);
    u.hash = "";
    return u.href.replace(/\/$/, "");
  } catch {
    return url.trim();
  }
};

const QUERY_STOP = new Set([
  "the",
  "and",
  "for",
  "are",
  "but",
  "not",
  "you",
  "all",
  "can",
  "has",
  "was",
  "what",
  "who",
  "how",
  "when",
  "where",
  "why",
  "about",
  "מה",
  "מי",
  "איך",
  "מתי",
  "איפה",
  "למה",
  "של",
  "על",
  "את",
  "זה",
  "היא",
  "הוא",
]);

export const tokenizeSearchQuery = (query: string): string[] => {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return [
    ...new Set(
      q
        .replace(/[^\p{L}\p{N}\s]/gu, " ")
        .split(/\s+/)
        .map((t) => t.trim())
        .filter((t) => t.length >= 2 && !QUERY_STOP.has(t)),
    ),
  ];
};

const hitSearchBlob = (hit: UnifiedSearchHit): string =>
  `${hit.titleOriginal ?? hit.title} ${hit.snippetOriginal ?? hit.snippet} ${hit.sourceLabel}`
    .toLowerCase()
    .trim();

const HEBREW_CHAR = /[\u0590-\u05FF]/;

export const isIsraeliHebrewRssHit = (hit: UnifiedSearchHit): boolean => {
  if (hit.kind !== "rss") return false;
  const key = (hit.sourceKey ?? "").toLowerCase();
  if (key.startsWith("il_")) return true;
  const title = hit.titleOriginal ?? hit.title;
  return HEBREW_CHAR.test(title) && HEBREW_CHAR.test(hit.sourceLabel);
};

/** Boost when title/snippet contains query terms — primary relevance signal for unified SERP. */
export const queryMatchBoost = (hit: UnifiedSearchHit, query: string, tokens?: string[]): number => {
  const terms = tokens ?? tokenizeSearchQuery(query);
  const q = query.trim().toLowerCase();
  const blob = hitSearchBlob(hit);
  const title = (hit.titleOriginal ?? hit.title).toLowerCase();
  let boost = 0;

  if (q.length >= 3 && blob.includes(q)) boost += 90;
  for (const term of terms) {
    if (title.includes(term)) boost += term.length >= 5 ? 45 : 28;
    else if (blob.includes(term)) boost += term.length >= 5 ? 22 : 12;
  }
  if (hit.provider === "wikipedia-en" || hit.provider === "wikipedia-he") {
    for (const term of terms) {
      if (title.includes(term)) boost += 35;
    }
  }
  return boost;
};

const effectiveScore = (
  hit: UnifiedSearchHit,
  query: string,
  tokens: string[],
  newsQuery: boolean,
  hebrewUi: boolean,
): number => {
  const boost = queryMatchBoost(hit, query, tokens);
  let base = hit.score ?? 0;
  const ilHeRss = hebrewUi && isIsraeliHebrewRssHit(hit);

  if (hit.kind === "rss") {
    if (!newsQuery && boost < (ilHeRss ? 10 : 18)) {
      base = Math.min(base, ilHeRss ? 52 : 28);
    }
    if (ilHeRss) base += 24;
  } else if (hit.kind === "hfmodel") {
    if (hit.meta?.hfStatus === "WORKING") base = Math.max(base, 72);
    else if (hit.meta?.hfStatus === "PROVIDER REQUIRED") base = Math.max(base, 58);
  } else if (
    hit.kind === "youtube" &&
    (shouldSearchYouTube(query) || isMusicQuery(query))
  ) {
    base = Math.max(base, 70);
  } else if (
    hit.kind === "video" &&
    hebrewUi &&
    (hit.sourceLabel === "Internet Archive" || hit.sourceLabel === "PeerTube")
  ) {
    if (HEBREW_CHAR.test(hit.titleOriginal ?? hit.title)) base += 20;
    if (/[\u0590-\u05FF]/.test(query)) base += 8;
  } else if (!newsQuery && (hit.kind === "web" || hit.provider === "wikipedia-en" || hit.provider === "wikipedia-he")) {
    base = Math.max(base, 38);
  }

  return base + boost;
};

/** Dedupe by URL, keep higher score. */
export const rankAndDedupeHits = (hits: UnifiedSearchHit[]): UnifiedSearchHit[] => {
  const byUrl = new Map<string, UnifiedSearchHit>();
  for (const hit of hits) {
    const key = normalizeUrl(hit.url);
    if (!key) continue;
    const prev = byUrl.get(key);
    if (!prev || (hit.score ?? 0) > (prev.score ?? 0)) {
      byUrl.set(key, hit);
    }
  }
  return [...byUrl.values()].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
};

/** Re-rank merged hits so query-relevant results (Wikipedia, matching web) rise to the top. */
export const rankHitsForQuery = (
  hits: UnifiedSearchHit[],
  query: string,
  options?: { newsQuery?: boolean; hebrewUi?: boolean },
): UnifiedSearchHit[] => {
  const q = query.trim();
  if (!q || !hits.length) return rankAndDedupeHits(hits);

  const tokens = tokenizeSearchQuery(q);
  const newsQuery = options?.newsQuery ?? false;
  const hebrewUi = options?.hebrewUi ?? false;
  const byUrl = new Map<string, UnifiedSearchHit>();

  for (const hit of hits) {
    const key = normalizeUrl(hit.url);
    if (!key) continue;
    const scored: UnifiedSearchHit = {
      ...hit,
      score: effectiveScore(hit, q, tokens, newsQuery, hebrewUi),
    };
    const prev = byUrl.get(key);
    if (!prev || (scored.score ?? 0) > (prev.score ?? 0)) {
      byUrl.set(key, scored);
    }
  }

  return [...byUrl.values()].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
};

export const filterHits = (
  hits: UnifiedSearchHit[],
  filter:
    | "all"
    | "rss"
    | "web"
    | "repos"
    | "movies"
    | "images"
    | "video"
    | "youtube"
    | "products"
    | "hfmodels",
): UnifiedSearchHit[] => {
  if (filter === "all") return hits;
  if (filter === "rss") return hits.filter((h) => h.kind === "rss");
  if (filter === "web") return hits.filter((h) => h.kind === "web");
  if (filter === "repos") return hits.filter((h) => h.kind === "github" || h.kind === "arxiv");
  if (filter === "movies") return hits.filter((h) => h.kind === "movie");
  if (filter === "images") return hits.filter((h) => h.kind === "image");
  if (filter === "video") return hits.filter((h) => h.kind === "video");
  if (filter === "youtube") return hits.filter((h) => h.kind === "youtube");
  if (filter === "products") return hits.filter((h) => h.kind === "product");
  if (filter === "hfmodels") return hits.filter((h) => h.kind === "hfmodel");
  return hits;
};
