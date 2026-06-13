import {
  classifySearchIntents,
  isFlightStatusQuery,
  isIssQuery,
  isMarketPriceQuery,
  isRedditQuery,
  isSatelliteCatalogQuery,
  isYouTubeQuery,
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
import { fetchNewsSearch } from "./providers/newsRss";
import { fetchAviationSearch } from "../realityData/providers/aviation";
import { fetchShipsSearch } from "../realityData/providers/ships";
import { fetchSatelliteCatalogSearch } from "../realityData/providers/satelliteCatalog";
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
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
import type { SearchSourceResult, WebSearchResult, WebSearchOptions } from "./types";

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
  return `חיפוש: ${ok.length} מקורות (${ok.map((s) => s.label).join(" · ")}) · ${intents.join(", ")}`;
};

export const formatWebSearchNoResultsContext = (): string =>
  `[WEB SEARCH — NO LIVE DATA]
The app tried live providers but none returned usable data for this question (timeout, CORS, or unsupported source).
RULES:
1. Say clearly in Hebrew that live data could not be loaded right now.
2. Do NOT invent numbers, prices, weather, places, repo names, or headlines.
3. Do NOT say you "cannot browse" — say the fetch failed or this data type is not supported in-browser yet.
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

/** Run routed parallel search — typically 1–5 s total. */
export const runWebSearch = async (query: string, options?: WebSearchOptions): Promise<WebSearchResult> => {
  const q = sanitizeSearchQuery(query);
  const intents = classifySearchIntents(q);
  const tasks: Promise<SearchSourceResult>[] = [];

  options?.onProgress?.({ type: "start", intents, query: q });

  if (intents.includes("worldtime")) tasks.push(trackTask(fetchWorldTimeSearch(q), options));
  if (intents.includes("weather")) tasks.push(trackTask(fetchWeatherSearch(q), options));
  if (intents.includes("marine")) tasks.push(trackTask(fetchMarineSearch(q), options));
  if (intents.includes("earthquake")) tasks.push(trackTask(fetchEarthquakeSearch(q), options));
  if (intents.includes("currency")) tasks.push(trackTask(fetchCurrencySearch(q), options));
  if (intents.includes("distance")) tasks.push(trackTask(fetchDistanceSearch(q), options));
  if (intents.includes("places")) tasks.push(trackTask(fetchPlacesSearch(q), options));
  if (intents.includes("ships")) tasks.push(trackTask(fetchShipsSearch(q), options));
  if (intents.includes("news")) tasks.push(trackTask(fetchNewsSearch(q), options));
  if (intents.includes("aviation")) tasks.push(trackTask(fetchAviationSearch(q, options?.recentUserText ?? []), options));
  if (intents.includes("satellite")) {
    if (isSatelliteCatalogQuery(q) || !isIssQuery(q)) {
      tasks.push(trackTask(fetchSatelliteCatalogSearch(q), options));
    }
    if (isIssQuery(q)) tasks.push(trackTask(fetchIssSearch(q), options));
  }
  if (intents.includes("spacex")) tasks.push(trackTask(fetchSpaceXLaunchSearch(q), options));
  if (intents.includes("spaceweather")) tasks.push(trackTask(fetchSpaceWeatherSearch(q), options));
  if (intents.includes("alerts")) tasks.push(trackTask(fetchIsraelAlertsSearch(q), options));
  if (intents.includes("disaster")) tasks.push(trackTask(fetchDisasterSearch(q), options));
  if (intents.includes("country")) tasks.push(trackTask(fetchCountrySearch(q), options));
  if (intents.includes("holiday")) tasks.push(trackTask(fetchHolidaySearch(q), options));
  if (intents.includes("government")) tasks.push(trackTask(fetchGovernmentSearch(q), options));
  if (intents.includes("github")) tasks.push(trackTask(fetchGitHubSearch(q), options));
  if (intents.includes("huggingface")) {
    tasks.push(trackTask(fetchHuggingFaceModelsSearch(q), options));
    tasks.push(trackTask(fetchHuggingFaceDatasetsSearch(q), options));
  }
  if (intents.includes("crypto")) tasks.push(trackTask(fetchCoinGeckoSearch(q), options));
  if (intents.includes("commodity")) tasks.push(trackTask(fetchCommoditySearch(q), options));
  if (intents.includes("market")) tasks.push(trackTask(fetchMarketQuoteSearch(q), options));
  if (intents.includes("hackernews")) tasks.push(trackTask(fetchHackerNewsSearch(q), options));
  if (isMarketPriceQuery(q) && !intents.includes("market")) {
    tasks.push(trackTask(fetchMarketQuoteSearch(q), options));
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
  if (isYouTubeQuery(q)) {
    tasks.push(
      trackTask(
        Promise.resolve(
          fetchUnsupportedSource(
            "youtube",
            "YouTube",
            "YouTube Data API דורש API key — לא מחובר בדפדפן; חפש ישירות ב-youtube.com או נסה שאלה על Hacker News / GitHub",
          ),
        ),
        options,
      ),
    );
  }
  if (intents.includes("wikipedia")) {
    const wikiQ = stripSearchVerb(q);
    tasks.push(trackTask(fetchWikipediaSearch(wikiQ, "en"), options));
    if (/[\u0590-\u05FF]/.test(q)) {
      tasks.push(trackTask(fetchWikipediaSearch(wikiQ, "he"), options));
    }
  }

  let settled = tasks.length ? await Promise.all(tasks) : [];

  const brief = buildSearchBrief(settled, intents, q);
  const okContext = formatSearchBriefContext(brief, q);
  const contextText = okContext.trim() && settled.some((s) => s.ok && s.text.trim())
    ? okContext
    : formatWebSearchNoResultsContext();

  options?.onProgress?.({ type: "complete", sources: settled });

  return {
    contextText,
    sources: settled,
    summaryHe: summarizeSearchResult(settled, intents),
    intents,
    brief,
  };
};

export const fetchWebContext = async (query: string): Promise<string> => {
  const result = await runWebSearch(query);
  return result.contextText;
};

export { userRequestsSearch } from "./intents";
