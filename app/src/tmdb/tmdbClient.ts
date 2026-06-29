import { getTmdbApiKey } from "../apiKeys/apiKeyStore";
import { isProviderEnabled, recordProviderUsage } from "../apiKeys/apiProviderUsage";
import type { EpgProgram } from "../liveMedia/epg/types";
import { tmdbFallbackLocale, type TmdbLanguage } from "./tmdbLocale";
import { shouldUseTmdbForEpg } from "./tmdbEpgGate";
import { localizeTmdbMetaForUi } from "./tmdbLocalize";

export type { TmdbLanguage } from "./tmdbLocale";
export { tmdbLocaleForUi } from "./tmdbLocale";

export type TmdbProgramMeta = {
  id: number;
  mediaType: "movie" | "tv";
  title: string;
  /** TV series name when title is an episode name. */
  seriesTitle?: string | null;
  runtimeMinutes: number | null;
  overview: string | null;
  posterUrl: string | null;
  rating: number | null;
  year: number | null;
};

const metaCache = new Map<string, TmdbProgramMeta | null>();

function cacheKey(title: string, language: TmdbLanguage, season?: number, episode?: number): string {
  return `${language}|${title.trim().toLowerCase()}|${season ?? ""}|${episode ?? ""}`;
}

function pickLocalizedText(
  primary: string | null | undefined,
  fallback: string | null | undefined,
): string | null {
  const p = primary?.trim();
  if (p) return p;
  const f = fallback?.trim();
  return f || null;
}

function posterUrl(path: string | null | undefined): string | null {
  if (!path?.trim()) return null;
  return `https://image.tmdb.org/t/p/w342${path}`;
}

function yearFromDate(d?: string): number | null {
  if (!d || d.length < 4) return null;
  const y = Number(d.slice(0, 4));
  return Number.isFinite(y) ? y : null;
}

async function tmdbFetch<T>(path: string, params: Record<string, string>): Promise<T | null> {
  const apiKey = getTmdbApiKey();
  if (!apiKey || !isProviderEnabled("tmdb")) return null;

  const qs = new URLSearchParams({ ...params, api_key: apiKey });
  const url = `https://api.themoviedb.org/3${path}?${qs}`;
  try {
    const res = await fetch(url);
    const text = await res.text();
    recordProviderUsage("tmdb", {
      ok: res.ok,
      hitCount: res.ok ? 1 : 0,
      bytesApprox: text.length,
    });
    if (!res.ok) return null;
    return JSON.parse(text) as T;
  } catch {
    recordProviderUsage("tmdb", { ok: false });
    return null;
  }
}

function lc(s?: string): string {
  return s?.trim().toLowerCase() ?? "";
}

type TmdbScored = { vote_count?: number; popularity?: number };

/** Order matches: exact title first, then by votes/popularity (avoids obscure 0-vote junk winning). */
function bestByRelevance<T extends TmdbScored>(matches: T[], isExact: (m: T) => boolean): T | null {
  if (!matches.length) return null;
  return [...matches].sort((a, b) => {
    const ax = isExact(a) ? 1 : 0;
    const bx = isExact(b) ? 1 : 0;
    if (ax !== bx) return bx - ax;
    const av = a.vote_count ?? 0;
    const bv = b.vote_count ?? 0;
    if (av !== bv) return bv - av;
    return (b.popularity ?? 0) - (a.popularity ?? 0);
  })[0];
}

export function pickMovie(
  results: Array<{
    id: number;
    title?: string;
    original_title?: string;
    release_date?: string;
    vote_average?: number;
    vote_count?: number;
    popularity?: number;
  }>,
  title: string,
) {
  const norm = lc(title);
  // Match the EPG title against both the localized and the original (English) title.
  const isExact = (r: { title?: string; original_title?: string }) =>
    lc(r.title) === norm || lc(r.original_title) === norm;
  const matches = results.filter(
    (r) => isExact(r) || lc(r.title).includes(norm) || lc(r.original_title).includes(norm),
  );
  const hit = bestByRelevance(matches, isExact);
  if (!hit) return null;
  // Reject obscure non-exact matches (e.g. a 0-vote home video that merely contains the title).
  if (!isExact(hit) && (hit.vote_count ?? 0) === 0 && (hit.popularity ?? 0) < 1) return null;
  const year = yearFromDate(hit.release_date);
  const wordCount = norm.split(/\s+/).filter(Boolean).length;
  if (wordCount >= 3 && year != null && year < 1965) return null;
  return hit;
}

export function pickTv(
  results: Array<{
    id: number;
    name?: string;
    original_name?: string;
    first_air_date?: string;
    vote_average?: number;
    vote_count?: number;
    popularity?: number;
  }>,
  title: string,
) {
  const norm = lc(title);
  // Localized name may differ from the EPG title (e.g. Hebrew UI) — also match original_name.
  const isExact = (r: { name?: string; original_name?: string }) =>
    lc(r.name) === norm || lc(r.original_name) === norm;
  const matches = results.filter(
    (r) => isExact(r) || lc(r.name).includes(norm) || lc(r.original_name).includes(norm),
  );
  return bestByRelevance(matches, isExact);
}

function resolveSearchTitle(title: string, channelTitle?: string): string {
  const t = title.trim();
  if (t) return t;
  return channelTitle?.trim() ?? "";
}

/** Movie / TV metadata for EPG now-playing (runtime, overview, poster). */
export async function lookupTmdbProgram(
  title: string,
  opts?: {
    season?: number;
    episode?: number;
    language?: TmdbLanguage;
    program?: EpgProgram | null;
    channelTitle?: string;
  },
): Promise<TmdbProgramMeta | null> {
  const stub: EpgProgram = opts?.program ?? {
    channelId: "",
    title,
    season: opts?.season,
    episode: opts?.episode,
    start: new Date(),
    end: new Date(),
  };
  if (!shouldUseTmdbForEpg(stub)) {
    return null;
  }

  const language = opts?.language ?? "en-US";
  const fallbackLanguage = tmdbFallbackLocale(language);
  const query = resolveSearchTitle(title, opts?.channelTitle);
  if (!query) return null;

  const key = `${cacheKey(query, language, opts?.season, opts?.episode)}|${opts?.channelTitle ?? ""}`;
  if (metaCache.has(key)) return metaCache.get(key) ?? null;
  if (!getTmdbApiKey()) {
    metaCache.set(key, null);
    return null;
  }

  const tvSearch = await tmdbFetch<{
    results?: Array<{
      id: number;
      name?: string;
      original_name?: string;
      first_air_date?: string;
      vote_average?: number;
      vote_count?: number;
      popularity?: number;
      overview?: string;
      poster_path?: string;
    }>;
  }>("/search/tv", { query, language, include_adult: "false" });

  const tvHit = pickTv(tvSearch?.results ?? [], query);

  if (tvHit?.id) {
    if (opts?.season != null && opts?.episode != null) {
      const ep = await tmdbFetch<{
        name?: string;
        overview?: string;
        runtime?: number;
        still_path?: string;
        vote_average?: number;
        air_date?: string;
      }>(`/tv/${tvHit.id}/season/${opts.season}/episode/${opts.episode}`, { language });

      const epEn =
        fallbackLanguage ?
          await tmdbFetch<{ name?: string; overview?: string }>(
            `/tv/${tvHit.id}/season/${opts.season}/episode/${opts.episode}`,
            { language: fallbackLanguage },
          )
        : null;

      const showBrief = await tmdbFetch<{ name?: string; poster_path?: string; first_air_date?: string }>(
        `/tv/${tvHit.id}`,
        { language },
      );

      if (ep) {
        const meta = await localizeTmdbMetaForUi(
          {
            id: tvHit.id,
            mediaType: "tv",
            seriesTitle: showBrief?.name?.trim() || tvHit.name?.trim() || query,
            title:
              pickLocalizedText(ep.name, epEn?.name) ||
              ep.name?.trim() ||
              tvHit.name?.trim() ||
              query,
            runtimeMinutes: ep.runtime && ep.runtime > 0 ? ep.runtime : null,
            overview: pickLocalizedText(ep.overview, epEn?.overview),
            posterUrl: posterUrl(ep.still_path ?? showBrief?.poster_path ?? tvHit.poster_path),
            rating: ep.vote_average ?? tvHit.vote_average ?? null,
            year: yearFromDate(ep.air_date ?? tvHit.first_air_date),
          },
          language,
        );
        metaCache.set(key, meta);
        return meta;
      }
    }

    const show = await tmdbFetch<{
      name?: string;
      overview?: string;
      poster_path?: string;
      vote_average?: number;
      first_air_date?: string;
      episode_run_time?: number[];
    }>(`/tv/${tvHit.id}`, { language });

    const showEn =
      fallbackLanguage ?
        await tmdbFetch<{ name?: string; overview?: string }>(`/tv/${tvHit.id}`, {
          language: fallbackLanguage,
        })
      : null;

    if (show) {
      const avgRuntime =
        show.episode_run_time?.length ?
          Math.round(show.episode_run_time.reduce((a, b) => a + b, 0) / show.episode_run_time.length)
        : null;

      const meta = await localizeTmdbMetaForUi(
        {
          id: tvHit.id,
          mediaType: "tv",
          title: pickLocalizedText(show.name, showEn?.name) || tvHit.name?.trim() || query,
          runtimeMinutes: avgRuntime,
          overview: pickLocalizedText(show.overview, showEn?.overview),
          posterUrl: posterUrl(show.poster_path),
          rating: show.vote_average ?? null,
          year: yearFromDate(show.first_air_date),
        },
        language,
      );
      metaCache.set(key, meta);
      return meta;
    }
  }

  const movieSearch = await tmdbFetch<{
    results?: Array<{
      id: number;
      title?: string;
      original_title?: string;
      release_date?: string;
      vote_average?: number;
      vote_count?: number;
      popularity?: number;
      overview?: string;
      poster_path?: string;
    }>;
  }>("/search/movie", { query, language, include_adult: "false" });

  const movieHit = pickMovie(movieSearch?.results ?? [], query);
  if (movieHit?.id) {
    const detail = await tmdbFetch<{
      title?: string;
      runtime?: number;
      overview?: string;
      poster_path?: string;
      vote_average?: number;
      release_date?: string;
    }>(`/movie/${movieHit.id}`, { language });

    const detailEn =
      fallbackLanguage ?
        await tmdbFetch<{
          title?: string;
          overview?: string;
        }>(`/movie/${movieHit.id}`, { language: fallbackLanguage })
      : null;

    if (detail) {
      const meta = await localizeTmdbMetaForUi(
        {
          id: movieHit.id,
          mediaType: "movie",
          title:
            pickLocalizedText(detail.title, detailEn?.title) ||
            movieHit.title?.trim() ||
            query,
          runtimeMinutes: detail.runtime && detail.runtime > 0 ? detail.runtime : null,
          overview: pickLocalizedText(detail.overview, detailEn?.overview),
          posterUrl: posterUrl(detail.poster_path),
          rating: detail.vote_average ?? null,
          year: yearFromDate(detail.release_date),
        },
        language,
      );
      metaCache.set(key, meta);
      return meta;
    }
  }

  metaCache.set(key, null);
  return null;
}

export async function lookupMovieRuntimeMinutes(title: string): Promise<number | null> {
  const meta = await lookupTmdbProgram(title);
  return meta?.runtimeMinutes ?? null;
}

export function resetTmdbCacheForTests(): void {
  metaCache.clear();
}
