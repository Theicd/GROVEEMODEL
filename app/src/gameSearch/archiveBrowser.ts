import { fetchJson } from "../webSearch/fetchJson";
import { filterBlacklistedGames, getBlacklistedIds } from "../localExperience/gamesStore";
import {
  buildOnlineArchiveQuery,
  buildTitleSearchQuery,
  FEATURED_ROTATION_POOL,
} from "./archiveQueries";
import {
  curatedForCategory,
  curatedMatchingQuery,
  mergeCurated,
} from "./curatedGames";
import { resolveGameSearch } from "./gameAliases";
import { nextRotationPage, pickRandomPage } from "./rotation";
import type { GameCategoryId, GameSearchResult, OnlineGame, ResolvedGameSearch } from "./types";

const ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php";
const PAGE_ROWS = 80;
const MAX_ROTATION_PAGES = 40;

/** Default page size for category browse in GamesPanel. */
export const GAMES_CATALOG_PAGE_SIZE = 20;

type ArchiveDoc = {
  identifier?: string;
  title?: string;
  description?: string | string[];
  emulator?: string;
  year?: string | number;
  downloads?: number;
  avg_rating?: number;
  num_reviews?: number;
};

type ArchiveResponse = {
  response?: { docs?: ArchiveDoc[]; numFound?: number };
};

type YearParams = {
  year?: number | null;
  yearFrom?: number | null;
  yearTo?: number | null;
};

function platformFromEmulator(emulator: string): string {
  const e = emulator.toLowerCase();
  if (e.includes("psx") || e === "ps1") return "PlayStation";
  if (e.includes("pcsx2") || e.includes("ps2")) return "PlayStation 2";
  if (e.includes("dosbox")) return "PC/DOS";
  if (e.includes("genesis") || e.includes("megadrive")) return "Genesis";
  if (e.includes("nes") || e.includes("nintendo")) return "NES";
  if (e.includes("snes")) return "SNES";
  if (e.includes("mame") || e.startsWith("arcade")) return "Arcade";
  return "Browser";
}

function parseYear(value: string | number | undefined): number | null {
  if (value === undefined || value === null) return null;
  const y = parseInt(String(value), 10);
  return y >= 1970 && y <= 2030 ? y : null;
}

function normalizeDoc(item: ArchiveDoc): OnlineGame | null {
  if (!item.identifier) return null;
  const id = item.identifier;
  const emulator = String(item.emulator ?? "").trim();
  if (!emulator) return null;

  const rawDesc = Array.isArray(item.description) ? item.description[0] : item.description ?? "";
  const description = String(rawDesc)
    .replace(/<[^>]+>/g, "")
    .trim()
    .slice(0, 200);

  return {
    id: `archive-${id}`,
    title: item.title || id,
    description,
    thumbnail: `https://archive.org/services/img/${id}`,
    playUrl: `https://archive.org/details/${id}`,
    embedUrl: `https://archive.org/embed/${id}`,
    source: "archive",
    gameType: "online",
    genre: "Classic",
    platform: platformFromEmulator(emulator),
    year: parseYear(item.year),
    downloads: typeof item.downloads === "number" ? item.downloads : undefined,
    rating: typeof item.avg_rating === "number" ? item.avg_rating : null,
    reviewsCount: typeof item.num_reviews === "number" ? item.num_reviews : undefined,
  };
}

function scoreTextMatch(game: OnlineGame, query: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 0;
  const title = game.title.toLowerCase();
  let score = 0;
  if (title === q) score += 100;
  if (title.startsWith(q)) score += 60;
  if (title.includes(q)) score += 40;
  if (game.description.toLowerCase().includes(q)) score += 10;
  const words = q.split(/\s+/).filter(Boolean);
  if (words.length > 1 && words.every((w) => title.includes(w))) score += 30;
  return score;
}

function qualityScore(game: OnlineGame, query: string, years?: YearParams): number {
  let score = scoreTextMatch(game, query);
  if (game.curated) score += 250;
  const dl = game.downloads ?? 0;
  if (dl > 0) score += Math.log10(dl + 1) * 18;
  const rating = game.rating ?? 0;
  if (rating > 0) score += rating * 12;
  const reviews = game.reviewsCount ?? 0;
  if (reviews > 0) score += Math.min(reviews, 20) * 2;
  if (game.year && years?.yearFrom != null && years?.yearTo != null) {
    if (game.year >= years.yearFrom && game.year <= years.yearTo) score += 25;
  }
  if (game.year && game.year >= 1975 && game.year <= 2005) score += 5;
  return score;
}

function dedupeGames(games: OnlineGame[]): OnlineGame[] {
  const seen = new Set<string>();
  return games.filter((g) => {
    const key = g.title.toLowerCase().trim();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function shuffle<T>(arr: T[]): T[] {
  return [...arr].sort(() => Math.random() - 0.5);
}

async function applyBrowseBlacklist(games: OnlineGame[]): Promise<OnlineGame[]> {
  const blacklist = await getBlacklistedIds();
  return filterBlacklistedGames(games, blacklist);
}

async function fetchArchiveDocs(
  q: string,
  rows: number,
  page = 1,
): Promise<{ games: OnlineGame[]; numFound: number }> {
  const params = new URLSearchParams({
    q,
    output: "json",
    rows: String(rows),
    page: String(page),
  });
  params.append("fl[]", "identifier");
  params.append("fl[]", "title");
  params.append("fl[]", "description");
  params.append("fl[]", "emulator");
  params.append("fl[]", "year");
  params.append("fl[]", "downloads");
  params.append("fl[]", "avg_rating");
  params.append("fl[]", "num_reviews");
  params.append("sort[]", "downloads desc");

  const url = `${ARCHIVE_SEARCH}?${params.toString()}`;
  const data = await fetchJson<ArchiveResponse>(url, undefined, { timeoutMs: 14_000 });
  const docs = data.response?.docs ?? [];
  const games = dedupeGames(docs.map(normalizeDoc).filter((g): g is OnlineGame => g !== null));
  return { games, numFound: data.response?.numFound ?? games.length };
}

async function fetchRotatedPool(
  q: string,
  rotationKey: string,
  pages = 2,
): Promise<OnlineGame[]> {
  const probe = await fetchArchiveDocs(q, PAGE_ROWS, 1);
  const totalPages = Math.max(1, Math.min(MAX_ROTATION_PAGES, Math.ceil(probe.numFound / PAGE_ROWS)));
  const pageA = pickRandomPage(rotationKey, totalPages);
  let pageB = pickRandomPage(`${rotationKey}-b`, totalPages);
  if (pageB === pageA && totalPages > 1) {
    pageB = (pageA % totalPages) + 1;
  }

  const fetches =
    pages <= 1
      ? [fetchArchiveDocs(q, PAGE_ROWS, pageA)]
      : [fetchArchiveDocs(q, PAGE_ROWS, pageA), fetchArchiveDocs(q, PAGE_ROWS, pageB)];

  const batches = await Promise.all(fetches);
  return dedupeGames(batches.flatMap((b) => b.games));
}

function effectiveCategory(category: GameCategoryId | null): GameCategoryId {
  if (category && category !== "featured") return category;
  const idx = nextRotationPage("featured-pool", FEATURED_ROTATION_POOL.length) - 1;
  return FEATURED_ROTATION_POOL[idx] ?? "arcade";
}

function queryParamsFromResolved(resolved: ResolvedGameSearch, category: GameCategoryId | null) {
  return {
    query: resolved.query,
    category,
    year: resolved.year,
    yearFrom: resolved.yearFrom,
    yearTo: resolved.yearTo,
  };
}

/** Search online-only games (Internet Archive, browser). */
export async function searchOnlineGames(
  query: string,
  limit = 12,
  category: GameCategoryId | null = null,
  year: number | null = null,
  yearFrom: number | null = null,
  yearTo: number | null = null,
): Promise<GameSearchResult> {
  const resolved = resolveGameSearch(query, category);
  return searchFromResolved(
    {
      ...resolved,
      query: resolved.query || query.trim(),
      year: resolved.year ?? year,
      yearFrom: resolved.yearFrom ?? yearFrom,
      yearTo: resolved.yearTo ?? yearTo,
      category: resolved.category ?? category,
    },
    limit,
  );
}

export async function searchFromResolved(
  resolved: ResolvedGameSearch,
  limit = 12,
): Promise<GameSearchResult> {
  const started = performance.now();
  const q = resolved.query.trim();
  const cat = resolved.category;
  const years: YearParams = {
    year: resolved.year,
    yearFrom: resolved.yearFrom,
    yearTo: resolved.yearTo,
  };
  const qp = queryParamsFromResolved(resolved, cat);

  let games: OnlineGame[] = [];

  if (q) {
    const primaryQ = buildOnlineArchiveQuery(qp);
    const primary = await fetchArchiveDocs(primaryQ, Math.min(limit * 6, 120));
    games = primary.games;

    if (!games.length) {
      const broadQ = buildTitleSearchQuery(q, resolved.year, resolved.yearFrom, resolved.yearTo);
      const broad = await fetchArchiveDocs(broadQ, Math.min(limit * 6, 120));
      games = broad.games;
    }

    if (!games.length && cat) {
      const catQ = buildOnlineArchiveQuery({ ...qp, year: null, yearFrom: null, yearTo: null });
      const catResult = await fetchArchiveDocs(catQ, Math.min(limit * 6, 120));
      games = catResult.games;
    }
  } else {
    const eff = effectiveCategory(cat);
    const poolQ = buildOnlineArchiveQuery({ ...qp, category: eff });
    games = await fetchRotatedPool(poolQ, `search-${eff}-${resolved.yearFrom ?? "all"}`, 2);
  }

  if (!games.length && resolved.yearFrom != null) {
    const noYearQ = buildOnlineArchiveQuery({
      ...qp,
      year: null,
      yearFrom: null,
      yearTo: null,
    });
    games = await fetchRotatedPool(noYearQ, `search-noyear-${cat ?? "all"}`, 2);
    if (resolved.yearTo != null) {
      const inEra = games.filter(
        (g) => g.year != null && g.year >= resolved.yearFrom! && g.year <= resolved.yearTo!,
      );
      if (inEra.length >= Math.min(limit, 4)) games = inEra;
    }
  }

  if (!games.length) {
    const fallbackCat = cat ?? (resolved.yearFrom != null && resolved.yearFrom < 1990 ? "retro" : "arcade");
    const poolQ = buildOnlineArchiveQuery({ ...qp, category: fallbackCat });
    games = await fetchRotatedPool(poolQ, `search-fb-${fallbackCat}`, 2);
  }

  const curated = q ? curatedMatchingQuery(q, 6) : curatedForCategory(cat);
  const sorted = [...mergeCurated(games, curated, limit * 3)].sort(
    (a, b) => qualityScore(b, q, years) - qualityScore(a, q, years),
  );
  const picked = sorted.slice(0, limit);
  const isExplicitSearch = q.length > 0 && !resolved.browseMode;

  if (q && !resolved.browseMode) {
    const matched = picked.filter((g) => scoreTextMatch(g, q) >= 30);
    if (!matched.length) {
      return {
        games: [],
        query: q,
        category: cat,
        latencyMs: Math.round(performance.now() - started),
        matchFound: false,
      };
    }
    return {
      games: matched,
      query: q,
      category: cat,
      latencyMs: Math.round(performance.now() - started),
      matchFound: true,
    };
  }

  const visible = isExplicitSearch ? picked : await applyBrowseBlacklist(picked);

  return {
    games: visible,
    query: q,
    category: cat,
    latencyMs: Math.round(performance.now() - started),
    matchFound: visible.length > 0,
  };
}

export async function randomOnlineGames(
  n = 8,
  category: GameCategoryId = "featured",
): Promise<GameSearchResult> {
  const started = performance.now();
  const eff = effectiveCategory(category);
  const poolQ = buildOnlineArchiveQuery({ category: eff });
  const pool = await fetchRotatedPool(poolQ, `random-${eff}`, 2);
  const curated = curatedForCategory(eff);
  const merged = mergeCurated(shuffle(pool), shuffle(curated), Math.max(n * 3, 24));
  const picked = shuffle(merged).slice(0, n);
  const fallback = picked.length ? picked : shuffle(curated).slice(0, n);
  const visible = await applyBrowseBlacklist(fallback);

  return {
    games: visible,
    query: "",
    category: eff,
    latencyMs: Math.round(performance.now() - started),
    matchFound: visible.length > 0,
  };
}

/** Fetch the next batch for category browse, skipping games already shown. */
export async function loadMoreOnlineGames(
  n: number,
  category: GameCategoryId,
  excludeIds: string[],
): Promise<GameSearchResult> {
  const started = performance.now();
  const eff = effectiveCategory(category);
  const poolQ = buildOnlineArchiveQuery({ category: eff });
  const blacklist = await getBlacklistedIds();
  const exclude = new Set([
    ...excludeIds.map((id) => id.toLowerCase()),
    ...[...blacklist].map((id) => id.toLowerCase()),
  ]);
  const picked: OnlineGame[] = [];
  let attempts = 0;

  while (picked.length < n && attempts < 5) {
    attempts += 1;
    const probe = await fetchArchiveDocs(poolQ, PAGE_ROWS, 1);
    const totalPages = Math.max(
      1,
      Math.min(MAX_ROTATION_PAGES, Math.ceil(probe.numFound / PAGE_ROWS)),
    );
    const page = pickRandomPage(`loadmore-${eff}`, totalPages);
    const batch = await fetchArchiveDocs(poolQ, PAGE_ROWS, page);
    const curated = curatedForCategory(eff);
    const merged = dedupeGames([...curated, ...batch.games]);
    for (const g of merged) {
      const key = g.id.toLowerCase();
      if (exclude.has(key)) continue;
      exclude.add(key);
      picked.push(g);
      if (picked.length >= n) break;
    }
  }

  return {
    games: picked,
    query: "",
    category: eff,
    latencyMs: Math.round(performance.now() - started),
    matchFound: picked.length > 0,
  };
}

export async function loadFeaturedFallback(): Promise<OnlineGame[]> {
  try {
    const base = import.meta.env.BASE_URL || "./";
    const normalized = base.endsWith("/") ? base : `${base}/`;
    const resp = await fetch(`${normalized}games/featured.json`);
    if (!resp.ok) return [];
    const data = (await resp.json()) as OnlineGame[];
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

export async function searchOnlineGamesWithFallback(
  resolved: ResolvedGameSearch,
  limit = 12,
): Promise<GameSearchResult> {
  const q = resolved.query.trim();

  if (resolved.browseMode && !q) {
    const browse = await randomOnlineGames(limit, resolved.category ?? "featured");
    return { ...browse, matchFound: browse.games.length > 0 };
  }

  try {
    const result = await searchFromResolved(resolved, limit);
    if (result.matchFound && result.games.length) return result;
    if (!q && result.games.length) return { ...result, matchFound: true };
  } catch {
    /* try curated fallback for title search */
  }

  if (q) {
    const curated = curatedMatchingQuery(q, limit);
    if (curated.length) {
      return {
        games: curated.slice(0, limit),
        query: q,
        category: resolved.category,
        latencyMs: 0,
        matchFound: true,
      };
    }
    return {
      games: [],
      query: q,
      category: resolved.category,
      latencyMs: 0,
      matchFound: false,
    };
  }

  if (resolved.browseMode) {
    const browse = await randomOnlineGames(limit, resolved.category ?? "featured");
    return { ...browse, matchFound: browse.games.length > 0 };
  }

  const fallback = await loadFeaturedFallback();
  const merged = mergeCurated(fallback, curatedForCategory(resolved.category), limit).slice(
    0,
    limit,
  );
  return {
    games: merged,
    query: q,
    category: resolved.category,
    latencyMs: 0,
    matchFound: merged.length > 0,
  };
}

export function archiveIdentifierFromGame(game: OnlineGame): string {
  return game.id.replace(/^archive-/, "");
}
