import { getTmdbApiKey } from "../apiKeys/apiKeyStore";
import { isProviderEnabled, recordProviderUsage } from "../apiKeys/apiProviderUsage";
import { tmdbFallbackLocale, type TmdbLanguage } from "./tmdbLocale";
import { localizeTmdbMetaForUi } from "./tmdbLocalize";

export type { TmdbLanguage } from "./tmdbLocale";
export { tmdbLocaleForUi } from "./tmdbLocale";

export type TmdbProgramMeta = {
  id: number;
  mediaType: "movie" | "tv";
  title: string;
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

function pickMovie(
  results: Array<{ id: number; title?: string; release_date?: string; vote_average?: number }>,
  title: string,
) {
  const norm = title.trim().toLowerCase();
  return (
    results.find((r) => r.title?.trim().toLowerCase() === norm) ??
    results.find((r) => r.title?.trim().toLowerCase().includes(norm)) ??
    results[0]
  );
}

function pickTv(
  results: Array<{ id: number; name?: string; first_air_date?: string; vote_average?: number }>,
  title: string,
) {
  const norm = title.trim().toLowerCase();
  return (
    results.find((r) => r.name?.trim().toLowerCase() === norm) ??
    results.find((r) => r.name?.trim().toLowerCase().includes(norm)) ??
    results[0]
  );
}

/** Movie / TV metadata for EPG now-playing (runtime, overview, poster). */
export async function lookupTmdbProgram(
  title: string,
  opts?: { season?: number; episode?: number; language?: TmdbLanguage },
): Promise<TmdbProgramMeta | null> {
  const language = opts?.language ?? "en-US";
  const fallbackLanguage = tmdbFallbackLocale(language);
  const key = cacheKey(title, language, opts?.season, opts?.episode);
  if (metaCache.has(key)) return metaCache.get(key) ?? null;
  if (!getTmdbApiKey()) {
    metaCache.set(key, null);
    return null;
  }

  const movieSearch = await tmdbFetch<{
    results?: Array<{
      id: number;
      title?: string;
      release_date?: string;
      vote_average?: number;
      overview?: string;
      poster_path?: string;
    }>;
  }>("/search/movie", { query: title.trim(), language, include_adult: "false" });

  const movieHit = pickMovie(movieSearch?.results ?? [], title);
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
            title,
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

  const tvSearch = await tmdbFetch<{
    results?: Array<{
      id: number;
      name?: string;
      first_air_date?: string;
      vote_average?: number;
      overview?: string;
      poster_path?: string;
    }>;
  }>("/search/tv", { query: title.trim(), language, include_adult: "false" });

  const tvHit = pickTv(tvSearch?.results ?? [], title);
  if (!tvHit?.id) {
    metaCache.set(key, null);
    return null;
  }

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

    if (ep) {
      const meta = await localizeTmdbMetaForUi(
        {
          id: tvHit.id,
          mediaType: "tv",
          title:
            pickLocalizedText(ep.name, epEn?.name) ||
            tvHit.name?.trim() ||
            title,
          runtimeMinutes: ep.runtime && ep.runtime > 0 ? ep.runtime : null,
          overview: pickLocalizedText(ep.overview, epEn?.overview),
          posterUrl: posterUrl(ep.still_path ?? tvHit.poster_path),
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

  if (!show) {
    metaCache.set(key, null);
    return null;
  }

  const avgRuntime =
    show.episode_run_time?.length ?
      Math.round(show.episode_run_time.reduce((a, b) => a + b, 0) / show.episode_run_time.length)
    : null;

  const meta = await localizeTmdbMetaForUi(
    {
      id: tvHit.id,
      mediaType: "tv",
      title: pickLocalizedText(show.name, showEn?.name) || tvHit.name?.trim() || title,
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

export async function lookupMovieRuntimeMinutes(title: string): Promise<number | null> {
  const meta = await lookupTmdbProgram(title);
  return meta?.runtimeMinutes ?? null;
}

export function resetTmdbCacheForTests(): void {
  metaCache.clear();
}
