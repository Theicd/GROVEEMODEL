/**
 * Parse Israeli cinema "now showing" listings from web search snippets.
 */
import type { SearchSourceResult, WebSerpHit } from "./types";

export type CinemaMovieHit = {
  title: string;
  note?: string;
  source: string;
  url?: string;
};

const NAV_JUNK =
  /^(?:ע(?:מוד)?\s*הבית|בקרוב בקולנוע|בדיק(?:ה|\/ביטול)|כרטיס(?:ים|יות)?|gift\s*card|vip|צור קשר|הופעות|סטנד(?:א)?פ|יום הקולנוע|hot cinema|cinema city|סינמה סיטי|עולם הילדים|הרצאות|סינמה נוסטלגיה|ארועים|כנסים|מהסרטים|onxy|все фильмы|המתחמי)/i;

const GENERIC_TITLE = /^(?:הסרט(?:-מתורגם(?:\s+ל(?:צרפתית|עברית|אנגלית))?)?|סרט|movie)$/i;

const HOME_TITLE =
  /(?:HOT CINEMA רשת|יום הקולנוע הישראלי|לא רק קולנוע|רשת בתי הקולנוע|סינמה סיטי\s*—\s*עמוד)/i;

const CINEMA_LISTING_URL =
  /(?:hotcinema\.co\.il\/(?:ShowingNow|movies)|cinema-city\.co\.il\/movies|seret\.co\.il\/movies|offscreen\.co\.il|mako\.co\.il.*cinema|walla\.co\.il.*cinema)/i;

const cleanTitle = (raw: string): string =>
  raw
    .replace(/^קופה ראשית:\s*/i, "")
    .replace(/\s*-\s*מדובב$/i, " (מדובב)")
    .replace(/\s+/g, " ")
    .trim();

const isUsableMovieTitle = (title: string): boolean => {
  if (title.length < 3 || title.length > 72) return false;
  if (NAV_JUNK.test(title)) return false;
  if (GENERIC_TITLE.test(title)) return false;
  if (/^(?:\d+\s*דק|$)/i.test(title)) return false;
  if (!/[\u0590-\u05FFa-zA-Z0-9]/.test(title)) return false;
  return true;
};

/** Extract movie titles from cinema-city / seret style snippets. */
export const parseCinemaMoviesFromText = (text: string): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();

  const add = (raw: string) => {
    const title = cleanTitle(raw);
    const key = title.toLowerCase();
    if (!isUsableMovieTitle(title) || seen.has(key)) return;
    seen.add(key);
    out.push(title);
  };

  const listingRe = /(?:^|[;؛])\s*(?:קופה ראשית:\s*)?([^·;]{2,80}?)\s*·\s*\d+/gim;
  for (const m of text.matchAll(listingRe)) {
    if (m[1]) add(m[1]);
  }

  const pipeRe = /(?:^|[|｜])\s*([\u0590-\u05FFa-zA-Z0-9][\u0590-\u05FFa-zA-Z0-9\s:.'\-]{2,60})\s*(?:\(|·|\||$)/g;
  if (out.length < 2) {
    for (const m of text.matchAll(pipeRe)) {
      if (m[1] && !/^\d+$/.test(m[1].trim())) add(m[1]);
    }
  }

  return out;
};

export const isCinemaHomepageHit = (hit: WebSerpHit): boolean => {
  const moviesInSnippet = parseCinemaMoviesFromText(hit.snippet ?? "").length;
  if (moviesInSnippet >= 2) return false;
  if (CINEMA_LISTING_URL.test(hit.url) && moviesInSnippet >= 1) return false;

  if (HOME_TITLE.test(hit.title)) return true;
  if (NAV_JUNK.test(hit.title.trim())) return true;
  if (/hotcinema\.co\.il\/?(?:#|$)/i.test(hit.url) && !/ShowingNow|movies/i.test(hit.url)) return true;
  if (/עמוד הבית/i.test(hit.title) && moviesInSnippet === 0) return true;
  if (/עכשיו בקולנוע/i.test(hit.title) && moviesInSnippet === 0 && !CINEMA_LISTING_URL.test(hit.url)) {
    return true;
  }
  return false;
};

export const extractCinemaMoviesFromSources = (
  sources: SearchSourceResult[],
  limit = 3,
): CinemaMovieHit[] => {
  const out: CinemaMovieHit[] = [];
  const seen = new Set<string>();

  const push = (title: string, source: string, url?: string, note?: string) => {
    const key = title.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ title, source, url, note });
  };

  const ordered = [...sources].sort((a, b) => {
    const score = (s: SearchSourceResult) => {
      let n = s.provider === "scavio" ? 30 : s.provider === "tavily" ? 10 : 0;
      for (const hit of s.webHits ?? []) {
        if (CINEMA_LISTING_URL.test(hit.url)) n += 40;
        n += parseCinemaMoviesFromText(hit.snippet ?? "").length * 8;
      }
      n += parseCinemaMoviesFromText(s.text).length * 5;
      return n;
    };
    return score(b) - score(a);
  });

  for (const src of ordered) {
    for (const hit of src.webHits ?? []) {
      if (isCinemaHomepageHit(hit)) continue;
      const fromSnippet = parseCinemaMoviesFromText(hit.snippet ?? "");
      if (fromSnippet.length) {
        for (const title of fromSnippet) {
          push(title, src.label, hit.url, "מוקרן כרגע בקולנוע");
          if (out.length >= limit) return out;
        }
        continue;
      }
      if (CINEMA_LISTING_URL.test(hit.url) && isUsableMovieTitle(hit.title)) {
        push(cleanTitle(hit.title), src.label, hit.url);
        if (out.length >= limit) return out;
      }
    }

    const lines = src.text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i] ?? "";
      const row = line.match(/^\d+\.\s+(.+)/);
      if (!row?.[1]) continue;
      const body = row[1];
      const urlMatch = body.match(/https?:\/\/\S+/);
      const url = urlMatch?.[0];
      const nextLine = (lines[i + 1] ?? "").trim();
      const snippetPart =
        nextLine.startsWith("   ") || /קופה ראשית:/.test(nextLine) ? nextLine : body.split(" · ").slice(1).join(" · ");
      const blob = `${body} ${snippetPart}`.trim();
      if (
        url &&
        !CINEMA_LISTING_URL.test(url) &&
        !/hotcinema|cinema-city|seret/i.test(url) &&
        parseCinemaMoviesFromText(blob).length === 0
      ) {
        continue;
      }

      const fromLine = parseCinemaMoviesFromText(blob);
      for (const title of fromLine) {
        push(title, src.label, url, "מוקרן כרגע בקולנוע");
        if (out.length >= limit) return out;
      }
    }
  }

  return out.slice(0, limit);
};

export const formatCinemaMovieBullets = (movies: CinemaMovieHit[], limit = 3): string[] =>
  movies.slice(0, limit).map((m) => {
    const note = m.note?.trim() || "מוקרן כרגע בבתי הקולנוע בישראל";
    return `• ${m.title} — ${note}`;
  });

export const wantsCinemaPlotSummaries = (query: string): boolean =>
  /(?:תקציר|summary|עליל(?:ה)?|plot|שורה אחת)/i.test(query);
