import {
  classifySearchIntents,
  isCrossSourceQuery,
  isFlightStatusQuery,
  isIssQuery,
  isMarketPriceQuery,
  isRedditQuery,
  isSatelliteCatalogQuery,
  isStarlinkCountQuery,
  isMoviesQuery,
  isLikelyMediaTitleQuery,
  isProductsQuery,
  sanitizeSearchQuery,
  stripSearchVerb,
} from "./intents";
import { fetchCurrencySearch } from "./providers/frankfurter";
import { fetchDistanceSearch } from "./providers/distance";
import { fetchEarthquakeSearch } from "./providers/usgsEarthquake";
import { fetchGitHubSearch } from "./providers/github";
import { fetchHuggingFaceDatasetsSearch, fetchHuggingFaceModelsSearch } from "./providers/huggingface";
import { fetchHolidaySearch } from "./providers/nagerHolidays";
import { fetchPlacesSearch } from "./providers/nominatimPlaces";
import { fetchGroveeNewsSearch } from "../groveeNews/newsToSearchBrief";
import { fetchAviationSearch } from "../realityData/providers/aviation";
import { fetchShipsSearch } from "../realityData/providers/ships";
import { fetchOverpassMarineSearch } from "./providers/overpassMarine";
import { fetchSatelliteCatalogSearch, fetchStarlinkCatalogSearch } from "../realityData/providers/satelliteCatalog";
import { fetchIssSearch } from "../realityData/providers/iss";
import { fetchSpaceWeatherSearch } from "../realityData/providers/spaceWeather";
import { fetchIsraelAlertsSearch } from "../realityData/providers/israelAlerts";
import { fetchDisasterSearch } from "../realityData/providers/disasters";
import { fetchMarineSearch } from "./providers/openMeteoMarine";
import { fetchWeatherSearch } from "./providers/openMeteo";
import { fetchCountrySearch } from "./providers/restCountries";
import { fetchWikipediaSearch } from "./providers/wikipedia";
import { fetchGovernmentSearch } from "./providers/wikidataGov";
import { fetchWorldTimeSearch } from "./providers/worldTime";
import { fetchCoinGeckoSearch } from "./providers/coingecko";
import { fetchCommoditySearch, fetchMarketQuoteSearch } from "./providers/marketQuotes";
import { fetchHackerNewsSearch } from "./providers/hackerNews";
import { fetchSpaceXLaunchSearch } from "./providers/spacexLaunch";
import { fetchUnsupportedSource } from "./providers/unsupported";
import { fetchSearxngSearch } from "./providers/searxng";
import { fetchAirQualitySearch } from "./providers/openMeteoAirQuality";
import { fetchArxivSearch } from "./providers/arxiv";
import { fetchUrlContextSearch } from "./providers/urlContext";
import { fetchMovieCatalogSearch } from "./providers/movieCatalog";
import {
  fetchPixabayImagesSearch,
  fetchPixabayVideosSearch,
} from "./providers/pixabayMedia";
import { fetchPeerTubeVideosSearch } from "./providers/peertubeMedia";
import { fetchInternetArchiveMediaSearch } from "./providers/internetArchiveMedia";
import { fetchInvidiousVideosSearch } from "./providers/invidiousMedia";
import { fetchIsraeliProductsSearch } from "./providers/israeliProducts";
import { applySnapshotFallbacks } from "../liveWorld/snapshotFallback";
import { pingGlobeForLiveSnapshot } from "../liveWorld/bridge";
import { buildCapabilityLiveReply, buildWebFallbackNoDataReply } from "./capabilityReplyMessages";
import { clearQueryCache } from "./queryCache";
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
import { validateLiveDataQuery } from "./entityValidation";
import { wrapWithQueryCache } from "./queryCache";
import { routeQuery, shouldAllowWebFallback } from "./routeQuery";
import { resolveSharedSearchRegion } from "./sharedRegion";
import { agentDebugLog } from "../debugAgentLog";
import { isStaticWebHost } from "./proxyFetch";
import type { SearchIntent, SearchProviderId, SearchSourceResult, WebSearchResult, WebSearchOptions } from "./types";

/** @deprecated Use formatSearchBriefContext via runWebSearch */
export const formatWebContext = (sources: SearchSourceResult[]): string => {
  const ok = sources.filter((s) => s.ok && s.text.trim());
  if (!ok.length) return "";
  const brief = buildSearchBrief(ok, [], "");
  return formatSearchBriefContext(brief, "", 800);
};

export const summarizeSearchResult = (sources: SearchSourceResult[], intents: string[]): string => {
  const ok = sources.filter((s) => s.ok);
  const failed = sources.filter((s) => !s.ok);
  if (!ok.length) {
    return failed.length
      ? `חיפוש: אין תוצאות (${failed.map((f) => f.label).join(", ")})`
      : "חיפוש: אין תוצאות";
  }
  const parts = [`${ok.length} מקורות (${ok.map((s) => s.label).join(" · ")})`];
  if (failed.length) {
    parts.push(`נכשל: ${failed.map((f) => f.label).join(" · ")}`);
  }
  return `חיפוש: ${parts.join(" · ")} · ${intents.join(", ")}`;
};

export const formatWebSearchNoResultsContext = (): string =>
  `[WEB SEARCH — NO LIVE DATA]
The app tried live providers but none returned usable data for this question (timeout, CORS, or unsupported source).
RULES:
1. Say clearly in Hebrew that live data could not be loaded right now (1–2 sentences).
2. Do NOT invent numbers, prices, weather, places, repo names, headlines, or politician names.
3. Do NOT say you "cannot browse" — say the fetch failed or this data type is not supported in-browser yet.
4. Do NOT philosophize, speculate, or ask the user to clarify — state what failed and stop.
5. Maximum 3 short sentences in Hebrew.
[/WEB SEARCH — NO LIVE DATA]`;

const trackTask = (
  task: Promise<SearchSourceResult>,
  options: WebSearchOptions | undefined,
): Promise<SearchSourceResult> => {
  if (!options?.onProgress) return task;
  return task.then((result) => {
    options.onProgress?.({ type: "provider_done", result });
    return result;
  });
};

const cached = (
  provider: SearchProviderId,
  query: string,
  fetch: () => Promise<SearchSourceResult>,
): Promise<SearchSourceResult> => wrapWithQueryCache(provider, query, fetch);

const mergeIntents = (a: SearchIntent[], b: SearchIntent[]): SearchIntent[] =>
  [...new Set([...a, ...b])];

const dedupeSources = (sources: SearchSourceResult[]): SearchSourceResult[] => {
  const byProvider = new Map<string, SearchSourceResult>();
  for (const s of sources) {
    const key = s.provider === "grovee-news" ? `${s.provider}:${s.label}` : s.provider;
    const prev = byProvider.get(key);
    if (!prev || (s.ok && !prev.ok)) byProvider.set(key, s);
  }
  return [...byProvider.values()];
};

/** PeerTube, Internet Archive video, and optional Invidious — in-browser playback. */
const pushFederatedVideoTasks = (
  tasks: Promise<SearchSourceResult>[],
  q: string,
  options?: WebSearchOptions,
) => {
  tasks.push(trackTask(cached("peertube-videos", q, () => fetchPeerTubeVideosSearch(q)), options));
  tasks.push(
    trackTask(cached("internet-archive-media", q, () => fetchInternetArchiveMediaSearch(q)), options),
  );
  tasks.push(trackTask(cached("invidious-videos", q, () => fetchInvidiousVideosSearch(q)), options));
};

/** YouTube-only search (Invidious mirror) — for music / explicit YouTube queries. */
const pushYouTubeTasks = (
  tasks: Promise<SearchSourceResult>[],
  q: string,
  options?: WebSearchOptions,
) => {
  tasks.push(trackTask(cached("invidious-videos", q, () => fetchInvidiousVideosSearch(q)), options));
};

const buildTasksForQuery = (
  q: string,
  intents: SearchIntent[],
  options?: WebSearchOptions,
): Promise<SearchSourceResult>[] => {
  const tasks: Promise<SearchSourceResult>[] = [];

  if (intents.includes("worldtime")) tasks.push(trackTask(cached("world-time", q, () => fetchWorldTimeSearch(q)), options));
  if (intents.includes("weather")) {
    tasks.push(
      trackTask(
        cached("open-meteo", q, () => fetchWeatherSearch(q, options?.sharedRegion?.place)),
        options,
      ),
    );
  }
  if (intents.includes("airquality")) {
    tasks.push(
      trackTask(
        cached("open-meteo-air-quality", q, () =>
          fetchAirQualitySearch(q, options?.sharedRegion?.place),
        ),
        options,
      ),
    );
  }
  if (intents.includes("arxiv")) tasks.push(trackTask(cached("arxiv", q, () => fetchArxivSearch(q)), options));
  if (intents.includes("marine")) tasks.push(trackTask(cached("open-meteo-marine", q, () => fetchMarineSearch(q, options?.sharedRegion?.place)), options));
  if (intents.includes("earthquake")) tasks.push(trackTask(cached("usgs-earthquake", q, () => fetchEarthquakeSearch(q)), options));
  if (intents.includes("currency")) tasks.push(trackTask(cached("frankfurter-fx", q, () => fetchCurrencySearch(q)), options));
  if (intents.includes("distance")) tasks.push(trackTask(cached("osrm-distance", q, () => fetchDistanceSearch(q)), options));
  if (intents.includes("places")) tasks.push(trackTask(cached("nominatim-places", q, () => fetchPlacesSearch(q)), options));
  if (intents.includes("ships")) tasks.push(trackTask(cached("ais-ships", q, () => fetchShipsSearch(q)), options));
  if (intents.includes("marine-infra")) {
    tasks.push(
      trackTask(
        cached("osm-overpass-marine", q, () => fetchOverpassMarineSearch(q)),
        options,
      ),
    );
  }
  if (intents.includes("news")) {
    tasks.push(
      trackTask(
        cached("grovee-news", q, () =>
          fetchGroveeNewsSearch(q, {
            onRefresh: (refreshed) => options?.onProgress?.({ type: "provider_done", result: refreshed }),
          }),
        ),
        options,
      ),
    );
  } else if (options?.plan?.blendNewsWithWeb) {
    tasks.push(
      trackTask(
        cached("grovee-news", q, () =>
          fetchGroveeNewsSearch(q, {
            onRefresh: (refreshed) => options?.onProgress?.({ type: "provider_done", result: refreshed }),
          }),
        ),
        options,
      ),
    );
  }
  if (intents.includes("aviation") || /\bawacs\b/i.test(q)) {
    tasks.push(
      trackTask(
        cached("adsb-aviation", q, () =>
          fetchAviationSearch(q, options?.recentUserText ?? []),
        ),
        options,
      ),
    );
  }
  if (intents.includes("satellite")) {
    if (isStarlinkCountQuery(q)) {
      tasks.push(trackTask(cached("starlink-catalog", q, () => fetchStarlinkCatalogSearch(q)), options));
    } else if (isSatelliteCatalogQuery(q) || !isIssQuery(q)) {
      tasks.push(trackTask(cached("celestrak", q, () => fetchSatelliteCatalogSearch(q)), options));
    }
    if (isIssQuery(q)) {
      pingGlobeForLiveSnapshot();
      tasks.push(trackTask(cached("iss-tracker", q, () => fetchIssSearch(q)), options));
    }
  }
  if (intents.includes("spacex")) tasks.push(trackTask(cached("spacex-launches", q, () => fetchSpaceXLaunchSearch(q)), options));
  if (intents.includes("spaceweather")) tasks.push(trackTask(cached("noaa-space", q, () => fetchSpaceWeatherSearch(q)), options));
  if (intents.includes("alerts")) tasks.push(trackTask(cached("israel-alerts", q, () => fetchIsraelAlertsSearch(q)), options));
  if (intents.includes("disaster")) tasks.push(trackTask(cached("gdacs-disasters", q, () => fetchDisasterSearch(q)), options));
  if (intents.includes("country")) tasks.push(trackTask(cached("rest-countries", q, () => fetchCountrySearch(q)), options));
  if (intents.includes("holiday")) tasks.push(trackTask(cached("nager-holidays", q, () => fetchHolidaySearch(q)), options));
  if (intents.includes("government")) tasks.push(trackTask(cached("wikidata-gov", q, () => fetchGovernmentSearch(q)), options));
  if (intents.includes("github")) tasks.push(trackTask(cached("github", q, () => fetchGitHubSearch(q)), options));
  if (intents.includes("movies")) {
    tasks.push(trackTask(cached("movie-catalog", q, () => fetchMovieCatalogSearch(q)), options));
    pushFederatedVideoTasks(tasks, q, options);
  }
  if (intents.includes("images")) {
    tasks.push(trackTask(cached("pixabay-images", q, () => fetchPixabayImagesSearch(q)), options));
  }
  if (intents.includes("video")) {
    tasks.push(trackTask(cached("pixabay-videos", q, () => fetchPixabayVideosSearch(q)), options));
    pushFederatedVideoTasks(tasks, q, options);
  }
  if (intents.includes("products")) {
    tasks.push(trackTask(cached("israeli-products", q, () => fetchIsraeliProductsSearch(q)), options));
  }
  if (intents.includes("huggingface")) {
    tasks.push(trackTask(cached("huggingface-models", q, () => fetchHuggingFaceModelsSearch(q)), options));
    tasks.push(trackTask(cached("huggingface-datasets", q, () => fetchHuggingFaceDatasetsSearch(q)), options));
  }
  if (intents.includes("crypto")) tasks.push(trackTask(cached("coingecko", q, () => fetchCoinGeckoSearch(q)), options));
  if (intents.includes("commodity")) tasks.push(trackTask(cached("stooq-commodity", q, () => fetchCommoditySearch(q)), options));
  if (intents.includes("market")) tasks.push(trackTask(cached("yahoo-finance", q, () => fetchMarketQuoteSearch(q)), options));
  if (intents.includes("hackernews")) tasks.push(trackTask(cached("hacker-news", q, () => fetchHackerNewsSearch(q)), options));
  if (isMarketPriceQuery(q) && !intents.includes("market")) {
    tasks.push(trackTask(cached("yahoo-finance", q, () => fetchMarketQuoteSearch(q)), options));
  }
  if (isRedditQuery(q)) {
    tasks.push(
      trackTask(
        Promise.resolve(
          fetchUnsupportedSource(
            "reddit",
            "Reddit",
            "Reddit API דורש OAuth — לא מחובר בדפדפן; נסה Hacker News או חיפוש כללי",
          ),
        ),
        options,
      ),
    );
  }
  if (isFlightStatusQuery(q)) {
    tasks.push(
      trackTask(
        Promise.resolve(
          fetchUnsupportedSource(
            "flight-status",
            "סטטוס טיסות",
            "סטטוס טיסות בנמל (JFK וכו') דורש AviationStack/FlightAware API — לא מחובר; בדוק אתר הנמל או FlightAware",
          ),
        ),
        options,
      ),
    );
  }
  if (intents.includes("youtube")) {
    pushYouTubeTasks(tasks, q, options);
  }
  if (intents.includes("wikipedia")) {
    const wikiQ = stripSearchVerb(q);
    tasks.push(trackTask(cached("wikipedia-en", wikiQ, () => fetchWikipediaSearch(wikiQ, "en")), options));
    if (/[\u0590-\u05FF]/.test(q)) {
      tasks.push(trackTask(cached("wikipedia-he", wikiQ, () => fetchWikipediaSearch(wikiQ, "he")), options));
    }
  }
  if (intents.includes("link")) {
    tasks.push(trackTask(cached("url-context", q, () => fetchUrlContextSearch(q)), options));
  }

  if (options?.panelSearch) {
    const wikiQ = stripSearchVerb(q) || q;
    if (!intents.includes("wikipedia")) {
      tasks.push(trackTask(cached("wikipedia-en", wikiQ, () => fetchWikipediaSearch(wikiQ, "en")), options));
      tasks.push(trackTask(cached("wikipedia-he", wikiQ, () => fetchWikipediaSearch(wikiQ, "he")), options));
    }
    if (!intents.includes("github")) {
      tasks.push(trackTask(cached("github", q, () => fetchGitHubSearch(q)), options));
    }
    if (!intents.includes("movies") && (isMoviesQuery(q) || isLikelyMediaTitleQuery(q))) {
      tasks.push(trackTask(cached("movie-catalog", q, () => fetchMovieCatalogSearch(q)), options));
    }
    if (!intents.includes("images")) {
      tasks.push(trackTask(cached("pixabay-images", q, () => fetchPixabayImagesSearch(q)), options));
    }
    if (!intents.includes("video")) {
      tasks.push(trackTask(cached("pixabay-videos", q, () => fetchPixabayVideosSearch(q)), options));
    }
    pushFederatedVideoTasks(tasks, q, options);
    if (!intents.includes("products") && isProductsQuery(q)) {
      tasks.push(trackTask(cached("israeli-products", q, () => fetchIsraeliProductsSearch(q)), options));
    }
    if (!intents.includes("huggingface")) {
      tasks.push(trackTask(cached("huggingface-models", q, () => fetchHuggingFaceModelsSearch(q)), options));
      tasks.push(trackTask(cached("huggingface-datasets", q, () => fetchHuggingFaceDatasetsSearch(q)), options));
    }
  }

  return tasks;
};

const runSingleQuerySearch = async (
  query: string,
  intents: SearchIntent[],
  options?: WebSearchOptions,
): Promise<{ sources: SearchSourceResult[]; intents: SearchIntent[] }> => {
  const q = sanitizeSearchQuery(query);
  const mergedIntents = mergeIntents(intents, classifySearchIntents(q));
  let tasks = buildTasksForQuery(q, mergedIntents, options);

  const wantWeb = shouldAllowWebFallback(tasks.length, options?.plan, q);

  if (wantWeb) {
    tasks.push(trackTask(cached("searxng", q, () => fetchSearxngSearch(q)), options));
  }

  let settled = tasks.length ? await Promise.all(tasks) : [];
  settled = applySnapshotFallbacks(q, mergedIntents, settled);
  return { sources: settled, intents: mergedIntents };
};

/** Run routed parallel search — typically 1–5 s total. */
export const runWebSearch = async (query: string, options?: WebSearchOptions): Promise<WebSearchResult> => {
  clearQueryCache();
  const q = sanitizeSearchQuery(query);
  const route = routeQuery(q, options?.plan);
  const intents = route.intents;
  const panelMode = options?.panelSearch === true;
  const effectiveRoute = panelMode
    ? {
        ...route,
        useWebFallback: true,
        blendNewsWithWeb: true,
        answerShape: route.answerShape === "short_fact" ? ("overview" as const) : route.answerShape,
      }
    : route;
  // #region agent log
  agentDebugLog("H1", "orchestrator.ts:runWebSearch", "route selected", { queryPreview: q.slice(0, 120), route: { intents, queries: effectiveRoute.queries, answerShape: effectiveRoute.answerShape, useWebFallback: effectiveRoute.useWebFallback, blendNewsWithWeb: effectiveRoute.blendNewsWithWeb } });
  // #endregion

  options?.onProgress?.({ type: "start", intents, query: q });

  const validation = validateLiveDataQuery(q, intents);
  if (!validation.ok) {
    options?.onProgress?.({ type: "complete", sources: [] });
    return {
      contextText: validation.contextText,
      sources: [],
      summaryHe: validation.summaryHe,
      intents,
      cannedReply: validation.cannedReply,
    };
  }

  const subQueries = effectiveRoute.queries;

  const sharedRegion = await resolveSharedSearchRegion(q, intents);
  const searchOptions: WebSearchOptions = {
    ...(sharedRegion ? { ...options, sharedRegion } : (options ?? {})),
    panelSearch: panelMode,
    plan: {
      ...options?.plan,
      useWebFallback: effectiveRoute.useWebFallback,
      blendNewsWithWeb: effectiveRoute.blendNewsWithWeb,
      answerShape: effectiveRoute.answerShape,
    },
  };

  const subResults = await Promise.all(
    subQueries.map((subQ) => runSingleQuerySearch(subQ, intents, searchOptions)),
  );

  const mergedIntents = subResults.reduce(
    (acc, r) => mergeIntents(acc, r.intents),
    intents,
  );
  const beforeDedupe = subResults.flatMap((r) => r.sources);
  let settled = dedupeSources(beforeDedupe);
  // #region agent log
  agentDebugLog("H3", "orchestrator.ts:runWebSearch", "sources before and after dedupe", {
    queryPreview: q.slice(0, 120),
    before: beforeDedupe.map((s) => ({ provider: s.provider, label: s.label, ok: s.ok, error: s.error?.slice(0, 120) })),
    after: settled.map((s) => ({ provider: s.provider, label: s.label, ok: s.ok, error: s.error?.slice(0, 120) })),
  });
  // #endregion

  const brief = buildSearchBrief(settled, mergedIntents, q, undefined, effectiveRoute.answerShape);
  const briefMax =
    isCrossSourceQuery(q) ? 1400 : mergedIntents.includes("news") ? (isStaticWebHost() ? 2200 : 1800) : 900;
  const okContext = formatSearchBriefContext(
    brief,
    q,
    briefMax,
    settled,
    effectiveRoute.answerShape,
    sharedRegion?.label,
  );
  const contextText = okContext.trim() && settled.some((s) => s.ok && s.text.trim())
    ? okContext
    : formatWebSearchNoResultsContext();

  options?.onProgress?.({ type: "complete", sources: settled });

  const cannedReply =
    buildCapabilityLiveReply(q, mergedIntents, settled, {
      answerShape: effectiveRoute.answerShape,
      regionLabel: sharedRegion?.label,
    }) ??
    (effectiveRoute.useWebFallback && !settled.some((s) => s.ok && s.text.trim())
      ? buildWebFallbackNoDataReply(q, settled)
      : null);

  return {
    contextText,
    sources: settled,
    summaryHe: summarizeSearchResult(settled, mergedIntents),
    intents: mergedIntents,
    brief,
    cannedReply,
  };
};

export const fetchWebContext = async (query: string): Promise<string> => {
  const result = await runWebSearch(query);
  return result.contextText;
};

export { userRequestsSearch } from "./intents";
export { warmLiveWorldCache } from "../liveWorld/fetchSnapshot";
export { clearQueryCache, queryCacheSize } from "./queryCache";
