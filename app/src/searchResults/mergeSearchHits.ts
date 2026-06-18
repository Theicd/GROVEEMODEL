import type { GroveeNewsCard } from "../groveeNews/types";

import type { MediaSerpHit, SearchProviderId, SearchSourceResult } from "../webSearch/types";

import { isNewsQuery, isMoviesQuery, isImagesQuery, isVideoMediaQuery, isProductsQuery, isHuggingFaceQuery, shouldSearchYouTube } from "../webSearch/intents";
import { isYouTubeUrl } from "../webSearch/youtubeUrls";

import { parseArxivText, parseGithubLines, parseHackerNewsLines, parseWikipediaText } from "./parseProviderLines";

import { getUserNewsProfile } from "../groveeNews/engine/settings/userNewsProfile";
import { rankAndDedupeHits, rankHitsForQuery } from "./rankHits";
import { filterBlockedHits } from "./serpBlocklist";

import { cleanDisplaySnippet } from "./snippetCleanup";

import { faviconForUrl, hostFromUrl } from "./sourceBranding";

import type { SearchResultsFacets, SearchResultsPayload, UnifiedSearchHit } from "./types";
import { youtubeHitFromMedia, youtubeHitFromWeb } from "./youtubeHits";

const mediaProviderId = (m: MediaSerpHit, kind: "image" | "video"): SearchProviderId => {
  switch (m.source) {
    case "Internet Archive":
      return "internet-archive-media";
    case "PeerTube":
      return "peertube-videos";
    case "Invidious":
      return "invidious-videos";
    default:
      return kind === "video" ? "pixabay-videos" : "pixabay-images";
  }
};

const isYoutubeMediaSource = (source?: string): boolean =>
  source === "YouTube" || source === "Invidious";

const mediaBaseScore = (m: MediaSerpHit, kind: "image" | "video" | "youtube"): number => {
  if (kind === "youtube") return 64;
  if (kind !== "video") return 50;
  if (m.source === "Internet Archive") return 66;
  if (m.source === "PeerTube") return 62;
  if (m.source === "Invidious") return 56;
  return 52;
};

export const newsCardToHit = (card: GroveeNewsCard, index: number): UnifiedSearchHit => {

  const snippet = cleanDisplaySnippet(card.title, card.summary?.trim() ?? "", card.url);

  return {

    id: card.id || `rss-${index}`,

    kind: "rss",

    title: card.title,

    titleOriginal: card.titleOriginal ?? card.title,

    url: card.url,

    snippet,

    snippetOriginal: card.summary?.trim() ?? "",

    imageUrl: card.image || undefined,

    sourceLabel: card.source,

    sourceKey: card.sourceKey,

    faviconUrl: faviconForUrl(card.url),

    provider: "grovee-news",

    publishedTs: card.publishedTs,

    score: Math.min(100, Math.max(0, card.score)),

    summarizable: true,

    meta: { engine: "RSS" },

  };

};



export const mergeSourcesToHits = (
  sources: SearchSourceResult[],
  query = "",
): UnifiedSearchHit[] => {

  const out: UnifiedSearchHit[] = [];



  for (const s of sources) {

    if (!s.ok) continue;



    if (s.newsCards?.length) {

      s.newsCards.forEach((card, i) => out.push(newsCardToHit(card, i)));

      continue;

    }



    if (s.webHits?.length) {

      for (const w of s.webHits) {
        const yt = youtubeHitFromWeb({
          id: w.id,
          title: w.title,
          url: w.url,
          snippet: w.snippet,
          score: 52,
        });
        if (yt) {
          out.push(yt);
          continue;
        }

        out.push({

          id: w.id,

          kind: "web",

          title: w.title,

          url: w.url,

          snippet: cleanDisplaySnippet(w.title, w.snippet, w.url),

          sourceLabel: hostFromUrl(w.url) || w.engine || "Web",

          faviconUrl: faviconForUrl(w.url),

          provider: "searxng",

          score: 45,

          meta: { engine: w.engine },

          summarizable: true,

        });

      }

      if (!s.mediaHits?.length) continue;

    }

    if (s.movieHits?.length) {
      for (const m of s.movieHits) {
        const displayTitle = m.year ? `${m.title} (${m.year})` : m.title;
        const metaLine = [
          m.ageRating,
          m.runtime ? `${m.runtime} דק'` : "",
          m.quality,
          m.rating != null ? `★${m.rating.toFixed(1)}` : "",
        ]
          .filter(Boolean)
          .join(" · ");

        if (!m.playUrl && isYouTubeUrl(m.url)) {
          const yt = youtubeHitFromWeb({
            id: m.id,
            title: displayTitle,
            url: m.url,
            snippet: metaLine ? `${m.snippet} · ${metaLine}` : m.snippet,
            score: 62 + Math.min(m.seeds ?? 0, 35),
          });
          if (yt) {
            out.push({
              ...yt,
              titleOriginal: m.originalTitle || m.title,
              imageUrl: m.poster || yt.imageUrl,
              provider: "movie-catalog",
              meta: metaLine
                ? { engine: metaLine, year: m.year, stars: m.rating }
                : m.rating != null
                  ? { year: m.year, stars: m.rating }
                  : { year: m.year },
            });
            continue;
          }
        }

        if (m.playUrl) {
          out.push({
            id: m.id,
            kind: "video",
            title: displayTitle,
            titleOriginal: m.originalTitle || m.title,
            url: m.url,
            snippet: cleanDisplaySnippet(displayTitle, m.snippet, m.url),
            snippetOriginal: m.snippet,
            imageUrl: m.poster,
            mediaPlayUrl: m.playUrl,
            durationSec: m.durationSec,
            sourceLabel: m.source || "Internet Archive",
            provider: "movie-catalog",
            score: 68 + Math.min(m.seeds ?? 0, 28),
            meta: metaLine ? { engine: metaLine, year: m.year, stars: m.rating } : { year: m.year },
            mediaEmbedMode: false,
            summarizable: false,
          });
          continue;
        }

        out.push({
          id: m.id,
          kind: "movie",
          title: displayTitle,
          titleOriginal: m.originalTitle || m.title,
          url: m.url,
          snippet: cleanDisplaySnippet(displayTitle, m.snippet, m.url),
          snippetOriginal: m.snippet,
          imageUrl: m.poster,
          sourceLabel: m.source === "TMDB" ? "TMDB" : m.source || "קטלוג",
          provider: "movie-catalog",
          score: 58 + Math.min(m.seeds ?? 0, 35),
          meta: metaLine
            ? { engine: metaLine, year: m.year, stars: m.rating }
            : m.rating != null
              ? { year: m.year, stars: m.rating }
              : { year: m.year },
          summarizable: false,
        });
      }
      continue;
    }

    if (s.mediaHits?.length) {
      for (const m of s.mediaHits) {
        if (isYoutubeMediaSource(m.source)) {
          out.push(
            youtubeHitFromMedia({
              id: m.id,
              title: m.title,
              url: m.url,
              snippet: m.snippet || m.tags,
              thumbnail: m.thumbnail,
              playUrl: m.playUrl,
              author: m.author,
              durationSec: m.durationSec,
              youtubeSubType: m.youtubeSubType ?? "video",
              score: mediaBaseScore(m, "youtube"),
              provider: "invidious-videos",
            }),
          );
          continue;
        }

        const kind = m.mediaType === "video" ? "video" : "image";
        out.push({
          id: m.id,
          kind,
          title: m.title,
          titleOriginal: m.title,
          url: m.licenseUrl || m.url,
          snippet: m.snippet?.trim() || m.tags || "",
          snippetOriginal: m.snippet || m.tags || "",
          imageUrl: m.thumbnail,
          mediaPlayUrl: m.playUrl,
          mediaEmbedMode: m.source === "Invidious",
          downloadUrl: m.downloadUrl,
          durationSec: m.durationSec,
          author: m.author,
          sourceLabel: m.source || "Pixabay",
          provider: mediaProviderId(m, kind),
          score: mediaBaseScore(m, kind),
          meta: m.durationSec ? { engine: `${m.durationSec}s` } : undefined,
          summarizable: false,
        });
      }
      continue;
    }

    if (s.productHits?.length) {
      for (const p of s.productHits) {
        const priceSnippet = p.priceSummary || p.snippet;
        out.push({
          id: p.id,
          kind: "product",
          title: p.title,
          titleOriginal: p.title,
          url: p.url,
          snippet: cleanDisplaySnippet(p.title, priceSnippet, p.url),
          snippetOriginal: priceSnippet,
          imageUrl: p.imageUrl,
          sourceLabel: p.source,
          provider: "israeli-products",
          score: p.priceNis != null ? 72 : 56,
          meta: { engine: p.barcode, priceNis: p.priceNis },
          summarizable: false,
        });
      }
      continue;
    }

    if (s.hfModelHits?.length) {
      for (const m of s.hfModelHits) {
        const statusBoost = m.status === "WORKING" ? 28 : m.probed ? 12 : 0;
        out.push({
          id: m.id,
          kind: "hfmodel",
          title: m.title,
          titleOriginal: m.modelId,
          url: m.url,
          snippet: cleanDisplaySnippet(m.title, m.snippet, m.url),
          snippetOriginal: m.snippet,
          imageUrl: "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
          sourceLabel: "Hugging Face",
          provider: "huggingface-models",
          score: 55 + statusBoost + Math.min(Math.floor((m.downloads ?? 0) / 50_000), 20),
          meta: {
            engine: m.category || m.pipelineTag || "Model",
            stars: m.likes,
            hfStatus: m.status,
            hfProvider: m.provider,
            hfAccess: m.accessMode,
            hfLatency: m.latency,
            hfCurl: m.curlSnippet,
            hfPython: m.pythonSnippet,
            hfCategory: m.category,
            hfPipeline: m.pipelineTag,
            hfProbeSource: m.probeSource,
          },
          summarizable: false,
        });
      }
      continue;
    }

    if (s.provider === "github" && s.text.trim()) {

      out.push(...parseGithubLines(s.text));

    } else if (s.provider === "hacker-news" && s.text.trim()) {

      out.push(...parseHackerNewsLines(s.text));

    } else if (s.provider === "arxiv" && s.text.trim()) {

      out.push(...parseArxivText(s.text));

    } else if (
      (s.provider === "wikipedia-en" || s.provider === "wikipedia-he") &&
      s.text.trim()
    ) {
      out.push(...parseWikipediaText(s.text, s.provider));
    }

  }



  const hebrewUi = getUserNewsProfile().uiLanguage === "he";
  return filterBlockedHits(
    rankHitsForQuery(rankAndDedupeHits(out), query, {
      newsQuery: isNewsQuery(query),
      hebrewUi,
    }),
  );
};



const buildFacets = (hits: UnifiedSearchHit[]): SearchResultsFacets => ({

  rss: hits.filter((h) => h.kind === "rss").length,

  web: hits.filter((h) => h.kind === "web").length,

  repos: hits.filter((h) => h.kind === "github").length,

  papers: hits.filter((h) => h.kind === "arxiv").length,

  movies: hits.filter((h) => h.kind === "movie").length,

  images: hits.filter((h) => h.kind === "image").length,

  videos: hits.filter((h) => h.kind === "video").length,

  youtube: hits.filter((h) => h.kind === "youtube").length,

  products: hits.filter((h) => h.kind === "product").length,

  hfModels: hits.filter((h) => h.kind === "hfmodel").length,

  other: hits.filter((h) => h.kind === "hackernews" || h.kind === "structured").length,

});



export const buildUnifiedSearchPayload = (

  query: string,

  sources: SearchSourceResult[],

): SearchResultsPayload => {

  const rawProviderErrors = sources
    .filter((s) => !s.ok && s.error)
    .map((s) => `${s.label}: ${s.error}`);

  const hits = mergeSourcesToHits(sources, query);

  const rssCount = hits.filter((h) => h.kind === "rss").length;

  const moviesCount = hits.filter((h) => h.kind === "movie").length;
  const imagesCount = hits.filter((h) => h.kind === "image").length;
  const videosCount = hits.filter((h) => h.kind === "video").length;
  const youtubeCount = hits.filter((h) => h.kind === "youtube").length;
  const productsCount = hits.filter((h) => h.kind === "product").length;
  const hfModelsCount = hits.filter((h) => h.kind === "hfmodel").length;

  /** Hide provider noise when we have results — but surface YouTube failures when the YouTube tab is empty. */
  const providerErrors =
    hits.length === 0
      ? rawProviderErrors
      : youtubeCount === 0
        ? rawProviderErrors.filter((e) => /YouTube|Invidious|Piped|SearXNG/i.test(e))
        : [];



  return {

    query,

    generatedAt: Date.now(),

    hits,

    facets: buildFacets(hits),

    providerErrors,

    preferRssFilter: isNewsQuery(query) && rssCount > 0,

    preferMoviesFilter: isMoviesQuery(query) && moviesCount > 0 && rssCount === 0,

    preferImagesFilter:
      isImagesQuery(query) && imagesCount > 0 && rssCount === 0 && moviesCount === 0,

    preferVideoFilter:
      (isVideoMediaQuery(query) || (isMoviesQuery(query) && videosCount > 0)) &&
      videosCount > 0 &&
      rssCount === 0 &&
      moviesCount === 0 &&
      youtubeCount === 0,

    preferYouTubeFilter:
      shouldSearchYouTube(query) &&
      youtubeCount > 0 &&
      rssCount === 0,

    preferProductsFilter:
      isProductsQuery(query) && productsCount > 0 && rssCount === 0 && moviesCount === 0,

    preferHfModelsFilter:
      isHuggingFaceQuery(query) && hfModelsCount > 0,

  };

};



export const shouldOpenSearchResultsPanel = (payload: SearchResultsPayload): boolean =>

  payload.hits.length > 0 || payload.providerErrors.some((e) => /SearXNG/i.test(e));

