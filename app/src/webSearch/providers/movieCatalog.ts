import { searchArchiveMovieHits } from "./internetArchiveSearch";
import { fetchJson } from "../fetchJson";
import { buildMoviesSearchQuery, isSeriesQuery } from "../intents";
import type { MovieSerpHit, SearchSourceResult } from "../types";

type WbEntityHit = {
  id: string;
  label: string;
  description?: string;
};

type WbSearchResponse = {
  search?: WbEntityHit[];
};

const WIKIDATA_HEADERS = {
  Accept: "application/json",
  "User-Agent": "GROVEEMODEL/1.0 (browser chat; movie catalog)",
};

const tmdbApiKey = (): string =>
  (import.meta.env.VITE_TMDB_API_KEY as string | undefined)?.trim() || "";

const stripHtml = (html: string): string => html.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

const parseWikidataYear = (time?: string): number | undefined => {
  if (!time) return undefined;
  const y = Number(String(time).slice(1, 5));
  return Number.isFinite(y) ? y : undefined;
};

const isFilmOrSeriesDescription = (desc: string): boolean =>
  /film|movie|television series|TV series|miniseries|directed by|סרט|סדרה|במאי/i.test(desc);

const normalizeTitleKey = (title: string): string =>
  title
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();

const dedupeMovieHits = (hits: MovieSerpHit[]): MovieSerpHit[] => {
  const byKey = new Map<string, MovieSerpHit>();
  for (const hit of hits) {
    const key = normalizeTitleKey(hit.originalTitle || hit.title);
    if (!key) continue;
    const prev = byKey.get(key);
    if (!prev || (hit.snippet?.length ?? 0) > (prev.snippet?.length ?? 0)) {
      byKey.set(key, hit);
    }
  }
  return [...byKey.values()];
};

async function enrichFromWikipedia(
  title: string,
  lang: "en" | "he",
): Promise<{ extract: string; poster?: string; url?: string }> {
  try {
    const searchUrl =
      `https://${lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(title)}` +
      `&srlimit=1&format=json&origin=*`;
    const searchData = await fetchJson<{
      query?: { search?: Array<{ title: string; pageid: number }> };
    }>(searchUrl, undefined, { timeoutMs: 8000 });
    const hit = searchData.query?.search?.[0];
    if (!hit) return { extract: "" };

    const detailUrl =
      `https://${lang}.wikipedia.org/w/api.php?action=query&prop=extracts|pageimages` +
      `&exintro=1&explaintext=1&piprop=thumbnail&pithumbsize=342&pageids=${hit.pageid}&format=json&origin=*`;
    const detail = await fetchJson<{
      query?: { pages?: Record<string, { extract?: string; thumbnail?: { source?: string }; title?: string }> };
    }>(detailUrl, undefined, { timeoutMs: 8000 });
    const page = Object.values(detail.query?.pages ?? {})[0];
    const pageTitle = page?.title ?? hit.title;
    const url = `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(pageTitle.replace(/ /g, "_"))}`;
    return {
      extract: (page?.extract ?? "").trim().slice(0, 480),
      poster: page?.thumbnail?.source,
      url,
    };
  } catch {
    return { extract: "" };
  }
}

async function fetchWikidataEntityBrief(qid: string): Promise<{
  year?: number;
  runtimeMin?: number;
  director?: string;
}> {
  try {
    const data = await fetchJson<{
      entities?: Record<
        string,
        {
          claims?: {
            P577?: Array<{ mainsnak?: { datavalue?: { value?: { time?: string } } } }>;
            P57?: Array<{ mainsnak?: { datavalue?: { value?: { id?: string } } } }>;
            P2047?: Array<{ mainsnak?: { datavalue?: { value?: { amount?: string } } } }>;
          };
        }
      >;
    }>(
      `https://www.wikidata.org/wiki/Special:EntityData/${qid}.json`,
      { headers: WIKIDATA_HEADERS },
      { timeoutMs: 9000 },
    );
    const ent = data.entities?.[qid];
    const year = parseWikidataYear(ent?.claims?.P577?.[0]?.mainsnak?.datavalue?.value?.time);
    const runtimeRaw = ent?.claims?.P2047?.[0]?.mainsnak?.datavalue?.value?.amount;
    const runtimeMin = runtimeRaw ? Math.round(Number(runtimeRaw)) : undefined;
    const directorQid = ent?.claims?.P57?.[0]?.mainsnak?.datavalue?.value?.id;
    let director: string | undefined;
    if (directorQid) {
      const dirData = await fetchJson<{
        entities?: Record<string, { labels?: { en?: { value?: string }; he?: { value?: string } } }>;
      }>(
        `https://www.wikidata.org/wiki/Special:EntityData/${directorQid}.json`,
        { headers: WIKIDATA_HEADERS },
        { timeoutMs: 7000 },
      );
      const labels = dirData.entities?.[directorQid]?.labels;
      director = labels?.he?.value || labels?.en?.value;
    }
    return { year, runtimeMin, director };
  } catch {
    return {};
  }
}

async function searchWikidataFilms(query: string): Promise<MovieSerpHit[]> {
  const langs = /[\u0590-\u05FF]/.test(query) ? (["he", "en"] as const) : (["en"] as const);
  const seen = new Set<string>();
  const out: MovieSerpHit[] = [];

  for (const lang of langs) {
    const data = await fetchJson<WbSearchResponse>(
      `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=${encodeURIComponent(query)}&language=${lang}&format=json&origin=*&limit=8&type=item`,
      { headers: WIKIDATA_HEADERS },
      { timeoutMs: 10_000 },
    );
    for (const item of data.search ?? []) {
      const desc = item.description ?? "";
      if (!isFilmOrSeriesDescription(desc)) continue;
      if (seen.has(item.id)) continue;
      seen.add(item.id);

      const meta = await fetchWikidataEntityBrief(item.id);
      const wikiLang = lang === "he" ? "he" : "en";
      const wiki = await enrichFromWikipedia(item.label, wikiLang);
      const yearFromDesc = desc.match(/\b(19|20)\d{2}\b/)?.[0];
      const year = meta.year ?? (yearFromDesc ? Number(yearFromDesc) : undefined);
      const snippet =
        wiki.extract ||
        desc ||
        [meta.director ? `במאי: ${meta.director}` : "", year ? `שנה: ${year}` : ""].filter(Boolean).join(" · ");

      out.push({
        id: `wikidata-${item.id}`,
        title: item.label,
        originalTitle: item.label,
        year,
        url: wiki.url || `https://www.wikidata.org/wiki/${item.id}`,
        snippet: snippet.slice(0, 480),
        poster: wiki.poster,
        runtime: meta.runtimeMin,
        source: "Wikidata",
      });
      if (out.length >= 6) return out;
    }
  }
  return out;
}

async function searchTvmaze(query: string): Promise<MovieSerpHit[]> {
  try {
    const data = await fetchJson<
      Array<{
        show?: {
          id: number;
          name: string;
          premiered?: string;
          summary?: string;
          image?: { medium?: string; original?: string };
          rating?: { average?: number };
          url?: string;
        };
      }>
    >(`https://api.tvmaze.com/search/shows?q=${encodeURIComponent(query)}`, undefined, { timeoutMs: 9000 });
    return (data ?? [])
      .slice(0, 5)
      .map((row, i) => {
        const show = row.show;
        if (!show) return null;
        const year = show.premiered ? Number(show.premiered.slice(0, 4)) : undefined;
        return {
          id: `tvmaze-${show.id ?? i}`,
          title: show.name,
          originalTitle: show.name,
          year: Number.isFinite(year) ? year : undefined,
          url: show.url || `https://www.tvmaze.com/shows/${show.id}`,
          snippet: stripHtml(show.summary ?? "").slice(0, 480),
          poster: show.image?.medium || show.image?.original,
          source: "TVMaze",
          rating: show.rating?.average,
        } as MovieSerpHit;
      })
      .filter((h): h is MovieSerpHit => h != null);
  } catch {
    return [];
  }
}

async function searchInternetArchive(query: string): Promise<MovieSerpHit[]> {
  try {
    return searchArchiveMovieHits(query, 8);
  } catch {
    return [];
  }
}

async function fetchTmdbFallback(query: string): Promise<MovieSerpHit[]> {
  const key = tmdbApiKey();
  if (!key) return [];
  try {
    const params = new URLSearchParams({ query, language: "he-IL", api_key: key });
    const data = await fetchJson<{
      results?: Array<{
        id: number;
        title?: string;
        original_title?: string;
        release_date?: string;
        overview?: string;
        poster_path?: string;
        vote_average?: number;
      }>;
    }>(`https://api.themoviedb.org/3/search/movie?${params}`, undefined, { timeoutMs: 9000 });
    return (data.results ?? []).slice(0, 5).map((m) => {
      const year = m.release_date ? Number(m.release_date.slice(0, 4)) : undefined;
      const title = m.title?.trim() || m.original_title?.trim() || "סרט";
      return {
        id: `tmdb-${m.id}`,
        title,
        originalTitle: m.original_title || m.title,
        year: Number.isFinite(year) ? year : undefined,
        url: `https://www.themoviedb.org/movie/${m.id}`,
        snippet: (m.overview?.trim() || "").slice(0, 480),
        poster: m.poster_path ? `https://image.tmdb.org/t/p/w342${m.poster_path}` : undefined,
        source: "TMDB",
        rating: m.vote_average,
      };
    });
  } catch {
    return [];
  }
}

const formatHitsText = (hits: MovieSerpHit[], query: string): string => {
  const lines = [`שאילתה: ${query}`];
  hits.forEach((h, i) => {
    const bits = [h.year, h.source, h.rating != null ? `★${h.rating.toFixed(1)}` : ""]
      .filter(Boolean)
      .join(" · ");
    lines.push(
      `${i + 1}. ${h.title}${bits ? ` (${bits})` : ""}${h.snippet ? `: ${h.snippet.slice(0, 160)}` : ""} (${h.url})`,
    );
  });
  return lines.join("\n");
};

export const fetchMovieCatalogSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "movie-catalog" as const;
  const label = "סרטים וסדרות";
  const mq = buildMoviesSearchQuery(query);
  if (!mq || mq.length < 2) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "הקלד שם סרט או סדרה לחיפוש (לפחות 2 תווים).",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const seriesHeavy = isSeriesQuery(query);
    const [wikidata, tvmaze, archive, tmdb] = await Promise.all([
      searchWikidataFilms(mq),
      searchTvmaze(mq),
      searchInternetArchive(mq),
      fetchTmdbFallback(mq),
    ]);

    const merged = dedupeMovieHits([
      ...wikidata,
      ...(seriesHeavy ? tvmaze : tvmaze.slice(0, 2)),
      ...tmdb,
      ...archive.slice(0, 6),
    ]).slice(0, 12);

    if (!merged.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצא מידע על «${mq}». נסה שם באנגלית (למשל Inception) או הוסף את המילה «סרט».`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const sources = [...new Set(merged.map((h) => h.source).filter(Boolean))].join(", ");
    return {
      provider,
      label,
      ok: true,
      text: `${formatHitsText(merged, mq)}\nמקורות: ${sources}`,
      url: merged[0]?.url,
      movieHits: merged,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: `שגיאה זמנית בחיפוש סרטים עבור «${mq}». נסה שוב בעוד רגע.`,
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
