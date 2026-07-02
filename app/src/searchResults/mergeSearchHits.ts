import type { GroveeNewsCard } from "../groveeNews/types";

import type { MediaSerpHit, SearchProviderId, SearchSourceResult } from "../webSearch/types";

import { isNewsQuery, isMoviesQuery, isImagesQuery, isVideoMediaQuery, isProductsQuery, isHuggingFaceQuery, shouldSearchYouTube, isMusicQuery, hasTimelyInfoSignal, isEarthquakeQuery, isDisasterQuery, isWeatherQuery, isMarineQuery, isAirQualityQuery, isPlacesQuery, isDistanceQuery, isShipsQuery, isMarineInfraQuery, isIsraelCinemaNowQuery } from "../webSearch/intents";
import { isYouTubeUrl } from "../webSearch/youtubeUrls";

import { parseArxivText, parseGithubLines, parseHackerNewsLines, parseWikipediaText } from "./parseProviderLines";

import { getUserNewsProfile } from "../groveeNews/engine/settings/userNewsProfile";
import { rankAndDedupeHits, rankHitsForQuery } from "./rankHits";
import { filterBlockedHits } from "./serpBlocklist";

import { cleanDisplaySnippet } from "./snippetCleanup";

import { webHitSourceLabel } from "./webProviderLabels";
import { faviconForUrl, hostFromUrl } from "./sourceBranding";

import type { SearchResultsFacets, SearchResultsPayload, UnifiedSearchHit } from "./types";
import { youtubeHitFromMedia, youtubeHitFromWeb } from "./youtubeHits";
import {
  formatLiveDisastersNote,
  mergeLiveDisasterHits,
  parseGdacsDisasterText,
  parseUsgsEarthquakeText,
} from "./liveDisastersHits";
import {
  formatLiveShipsNote,
  mergeLiveShipHits,
  parseAisShipsText,
  parseMarineInfraText,
  shipsPanelTotal,
} from "./liveShipsHits";
import { getLiveWorldSnapshotForPanel } from "../liveWorld/snapshotStore";
import { parseStructuredProviderHits } from "./structuredProviderHits";
import { isCinemaHomepageHit, parseCinemaMoviesFromText } from "../webSearch/cinemaIlExtract";
import { buildWebTopicSearchPlan } from "../webSearch/webTopicQueryPlan";

const mediaProviderId = (m: MediaSerpHit, kind: "image" | "video"): SearchProviderId => {
  if (m.source?.startsWith("OpenSERP")) return "openserp";
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

    if (!s.ok) {
      if (s.provider === "grovee-news" && s.newsCards?.length) {
        s.newsCards.forEach((card, i) => out.push(newsCardToHit(card, i)));
      }
      continue;
    }



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
          out.push({
            ...yt,
            provider:
              s.provider === "openserp"
                ? "openserp"
                : s.provider === "tavily"
                  ? "tavily"
                  : s.provider === "scavio"
                    ? "scavio"
                    : yt.provider,
            sourceLabel: s.provider === "openserp" ? "YouTube · OpenSERP" : yt.sourceLabel,
          });
          continue;
        }

        out.push({

          id: w.id,

          kind: "web",

          title: w.title,

          url: w.url,

          snippet: cleanDisplaySnippet(w.title, w.snippet, w.url),

          sourceLabel: webHitSourceLabel(
            s.provider === "tavily"
              ? "tavily"
              : s.provider === "scavio"
                ? "scavio"
                : s.provider === "openserp"
                  ? "openserp"
                  : "searxng",
            w.url,
            w.engine,
          ),

          faviconUrl: faviconForUrl(w.url),

          provider:
            s.provider === "tavily"
              ? "tavily"
              : s.provider === "scavio"
                ? "scavio"
                : s.provider === "openserp"
                  ? "openserp"
                  : "searxng",

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

    if (s.liveMediaHits?.length) {
      for (const m of s.liveMediaHits) {
        const kind = m.mediaType === "radio" ? "radio" : "livetv";
        const statusBoost = m.status === "working" ? 24 : m.status === "unknown" ? 6 : 0;
        const fuseBoost = Math.round((m.fuseScore ?? 0) * 40);
        out.push({
          id: m.id,
          kind,
          title: m.title,
          titleOriginal: m.title,
          url: m.url,
          snippet: m.snippet?.trim() || m.category || "",
          snippetOriginal: m.snippet || "",
          imageUrl: m.logoUrl,
          mediaPlayUrl: m.streamUrl,
          sourceLabel: kind === "radio" ? "Radio" : "TV LIVE",
          provider: "live-tv",
          score: (kind === "radio" && isMusicQuery(query) ? 68 : 58) + statusBoost + fuseBoost,
          meta: {
            engine: m.category || m.codec || "Live",
            year: m.bitrate,
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
    } else if (s.provider === "usgs-earthquake" && s.text.trim()) {
      out.push(...parseUsgsEarthquakeText(s.text));
    } else if (s.provider === "gdacs-disasters" && s.text.trim()) {
      out.push(...parseGdacsDisasterText(s.text));
    } else if (s.provider === "ais-ships" && s.text.trim()) {
      out.push(...parseAisShipsText(s.text));
    } else if (s.provider === "osm-overpass-marine" && s.text.trim()) {
      out.push(...parseMarineInfraText(s.text));
    } else if (
      s.text.trim() &&
      (s.provider === "open-meteo" ||
        s.provider === "open-meteo-marine" ||
        s.provider === "open-meteo-air-quality" ||
        s.provider === "nominatim-places" ||
        s.provider === "osrm-distance")
    ) {
      out.push(...parseStructuredProviderHits(s));
    }

  }



  const hebrewUi = getUserNewsProfile().uiLanguage === "he";
  const topicPlan = buildWebTopicSearchPlan(query);
  if (topicPlan?.kind === "cinema_il") {
    for (const hit of out) {
      if (hit.kind !== "web") continue;
      const snippet = hit.snippetOriginal ?? hit.snippet;
      const movies = parseCinemaMoviesFromText(snippet);
      if (movies.length >= 2) {
        hit.score = (hit.score ?? 0) + 220;
      } else if (
        isCinemaHomepageHit({
          title: hit.titleOriginal ?? hit.title,
          url: hit.url,
          snippet,
        })
      ) {
        hit.score = Math.max(0, (hit.score ?? 0) - 180);
      }
    }
  }
  const rankQuery = query;
  const ranked = filterBlockedHits(
    rankHitsForQuery(rankAndDedupeHits(out), rankQuery, {
      newsQuery: isNewsQuery(query) && !topicPlan,
      hebrewUi,
    }),
  );
  if (topicPlan) return ranked;
  return mergeLiveShipHits(mergeLiveDisasterHits(ranked, query), query);
};



const buildFacets = (hits: UnifiedSearchHit[]): SearchResultsFacets => ({

  rss: hits.filter((h) => h.kind === "rss").length,

  web: hits.filter((h) => h.kind === "web" && h.provider !== "openserp").length,

  companionWeb: hits.filter((h) => h.kind === "web" && h.provider === "openserp").length,

  repos: hits.filter((h) => h.kind === "github").length,

  papers: hits.filter((h) => h.kind === "arxiv").length,

  movies: hits.filter((h) => h.kind === "movie").length,

  images: hits.filter((h) => h.kind === "image").length,

  videos: hits.filter((h) => h.kind === "video").length,

  youtube: hits.filter((h) => h.kind === "youtube").length,

  liveTv: hits.filter((h) => h.kind === "livetv").length,

  radio: hits.filter((h) => h.kind === "radio").length,

  products: hits.filter((h) => h.kind === "product").length,

  hfModels: hits.filter((h) => h.kind === "hfmodel").length,

  earthquakes: hits.filter((h) => h.kind === "earthquake").length,

  disasters: hits.filter((h) => h.kind === "disaster").length,

  ships: hits.filter((h) => h.kind === "ship" || h.kind === "marine").length,

  weather: hits.filter((h) => h.kind === "weather").length,

  marine: hits.filter((h) => h.kind === "marine").length,

  places: hits.filter((h) => h.kind === "place" || h.kind === "route").length,

  other: hits.filter((h) => h.kind === "hackernews" || h.kind === "structured").length,

});



export const buildUnifiedSearchPayload = (

  query: string,

  sources: SearchSourceResult[],

): SearchResultsPayload => {

  const topicPlan = buildWebTopicSearchPlan(query);
  const rawProviderErrors = sources
    .filter((s) => !s.ok && s.error)
    .map((s) => `${s.label}: ${s.error}`);

  const hits = mergeSourcesToHits(sources, query);

  const companionWebCount = hits.filter((h) => h.kind === "web" && h.provider === "openserp").length;
  const openserpSource = sources.find((s) => s.provider === "openserp");

  const webCount = hits.filter((h) => h.kind === "web" && h.provider !== "openserp").length;

  const rssCount = hits.filter((h) => h.kind === "rss").length;

  const moviesCount = hits.filter((h) => h.kind === "movie").length;
  const imagesCount = hits.filter((h) => h.kind === "image").length;
  const videosCount = hits.filter((h) => h.kind === "video").length;
  const youtubeCount = hits.filter((h) => h.kind === "youtube").length;
  const liveTvCount = hits.filter((h) => h.kind === "livetv").length;
  const radioCount = hits.filter((h) => h.kind === "radio").length;
  const productsCount = hits.filter((h) => h.kind === "product").length;
  const hfModelsCount = hits.filter((h) => h.kind === "hfmodel").length;
  const earthquakesCount = hits.filter((h) => h.kind === "earthquake").length;
  const disastersCount = hits.filter((h) => h.kind === "disaster").length;
  const shipsCount = shipsPanelTotal(hits);
  const weatherCount = hits.filter((h) => h.kind === "weather").length;

  /** Hide provider noise when we have results — but surface YouTube failures when the YouTube tab is empty. */
  const providerErrors =
    hits.length === 0
      ? rawProviderErrors
      : rssCount === 0 && rawProviderErrors.some((e) => /GROVEE NEWS|חדשות/i.test(e))
        ? rawProviderErrors.filter((e) => /GROVEE NEWS|חדשות|YouTube|Invidious|SearXNG|TV LIVE|Radio|live-tv|קטלוג/i.test(e))
        : youtubeCount === 0 || (liveTvCount === 0 && radioCount === 0)
          ? rawProviderErrors.filter((e) =>
              /YouTube|Invidious|Piped|SearXNG|TV LIVE|Radio|live-tv|קטלוג/i.test(e),
            )
          : [];



  const newsSource = sources.find((s) => s.provider === "grovee-news");
  const newsRssNote = newsSource?.newsScanNote;
  const snapshot = getLiveWorldSnapshotForPanel();
  const liveDisastersNote = formatLiveDisastersNote(snapshot, "he");
  const liveShipsNote = formatLiveShipsNote(snapshot, "he", query);

  const wantsEarthquakeTab = isEarthquakeQuery(query);
  const wantsDisasterTab = isDisasterQuery(query) && !wantsEarthquakeTab;
  const wantsLiveSensorTab = wantsEarthquakeTab || wantsDisasterTab;
  const wantsWeatherTab =
    isWeatherQuery(query) || isMarineQuery(query) || isAirQualityQuery(query);
  const wantsShipsTab =
    !topicPlan && (isShipsQuery(query) || isMarineInfraQuery(query));
  const wantsPlacesTab = isPlacesQuery(query) || isDistanceQuery(query);
  const wantsEventsTab =
    /אירוע|חריג|מה קורה בעולם|current events/i.test(query) ||
    wantsEarthquakeTab ||
    wantsDisasterTab ||
    wantsWeatherTab ||
    (earthquakesCount > 0 && disastersCount > 0);
  const openMeteoMarineCount = hits.filter(
    (h) => h.kind === "marine" && h.provider === "open-meteo-marine",
  ).length;
  const eventsCount =
    earthquakesCount + disastersCount + weatherCount + openMeteoMarineCount;

  return {

    query,

    generatedAt: Date.now(),

    hits,

    facets: buildFacets(hits),

    providerErrors,

    newsRssNote,

    liveDisastersNote,

    liveShipsNote,

    preferShipsFilter:
      shipsCount > 0 &&
      wantsShipsTab &&
      rssCount === 0 &&
      !wantsLiveSensorTab &&
      !wantsEventsTab,

    preferEventsFilter: eventsCount > 0 && wantsEventsTab && rssCount === 0,

    preferRssFilter:
      !topicPlan &&
      !wantsLiveSensorTab &&
      !wantsWeatherTab &&
      !wantsPlacesTab &&
      !wantsShipsTab &&
      rssCount > 0 &&
      (isNewsQuery(query) || hasTimelyInfoSignal(query) || (rssCount >= 2 && youtubeCount === 0 && liveTvCount === 0)),

    showCompanionTab: !!openserpSource,

    companionWebError:
      openserpSource && !openserpSource.ok ? openserpSource.error : undefined,

    preferMoviesFilter:
      (isMoviesQuery(query) || isIsraelCinemaNowQuery(query)) &&
      moviesCount > 0 &&
      rssCount === 0 &&
      !topicPlan,

    preferImagesFilter:
      isImagesQuery(query) && imagesCount > 0 && rssCount === 0 && moviesCount === 0,

    preferVideoFilter:
      (isVideoMediaQuery(query) ||
        (isMoviesQuery(query) && videosCount > 0) ||
        shouldSearchYouTube(query)) &&
      videosCount + youtubeCount > 0 &&
      rssCount === 0 &&
      moviesCount === 0,

    preferProductsFilter: isProductsQuery(query) && productsCount > 0,

    preferHfModelsFilter:
      isHuggingFaceQuery(query) && hfModelsCount > 0,

  };

};



/** Chat turns keep search inline; the side panel opens only from the rail (manual). */
export const shouldOpenSearchResultsPanel = (_payload: SearchResultsPayload): boolean => false;

