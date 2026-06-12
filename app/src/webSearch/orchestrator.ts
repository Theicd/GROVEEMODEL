import { classifySearchIntents, stripSearchVerb } from "./intents";

import { fetchCurrencySearch } from "./providers/frankfurter";

import { fetchDistanceSearch } from "./providers/distance";

import { fetchEarthquakeSearch } from "./providers/usgsEarthquake";

import { fetchGitHubSearch } from "./providers/github";

import {

  fetchHuggingFaceDatasetsSearch,

  fetchHuggingFaceModelsSearch,

} from "./providers/huggingface";

import { fetchHolidaySearch } from "./providers/nagerHolidays";

import { fetchPlacesSearch } from "./providers/nominatimPlaces";

import { fetchNewsSearch } from "./providers/newsRss";

import { fetchAviationSearch } from "../realityData/providers/aviation";
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

import { fetchSearxSearch } from "./providers/searxng";
import { fetchRedditSearch } from "./providers/reddit";
import { fetchHackerNewsSearch } from "./providers/hackernews";
import { fetchArxivSearch } from "./providers/arxiv";
import { fetchCoinGeckoSearch } from "./providers/coingecko";

import type { SearchSourceResult, WebSearchResult, WebSearchOptions, SearchIntent } from "./types";



const GROUNDING_HEADER = `[WEB SEARCH RESULTS — authoritative live data. Use ONLY this block for factual answers.

If data is missing, say so clearly. Cite source names. Do NOT invent numbers or URLs.]`;



export const formatWebContext = (sources: SearchSourceResult[]): string => {

  const ok = sources.filter((s) => s.ok && s.text.trim());

  if (!ok.length) return "";

  const blocks = ok.map((s) => `## ${s.label}\n${s.text}${s.url ? `\nSource: ${s.url}` : ""}`);

  return `${GROUNDING_HEADER}\n\n${blocks.join("\n\n")}\n\n[/WEB SEARCH RESULTS]`;

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

const STRUCTURED_INTENTS: SearchIntent[] = [
  "worldtime", "weather", "marine", "earthquake", "currency", "holiday", "government",
  "country", "distance", "places", "news", "aviation", "satellite", "spaceweather",
  "alerts", "disaster", "market", "reddit", "hackernews", "arxiv",
];

const runWikiSearxFallback = async (query: string): Promise<SearchSourceResult[]> => {
  const wikiQ = stripSearchVerb(query);
  const tasks: Promise<SearchSourceResult>[] = [
    fetchWikipediaSearch(wikiQ, "en"),
    fetchSearxSearch(query),
  ];
  if (/[\u0590-\u05FF]/.test(query)) {
    tasks.push(fetchWikipediaSearch(wikiQ, "he"));
  }
  return Promise.all(tasks);
};

/** Run routed parallel search — typically 1–5 s total. */
export const runWebSearch = async (query: string, options?: WebSearchOptions): Promise<WebSearchResult> => {

  const q = query.trim();

  const intents = classifySearchIntents(q);

  const tasks: Promise<SearchSourceResult>[] = [];



  if (intents.includes("worldtime")) tasks.push(fetchWorldTimeSearch(q));

  if (intents.includes("weather")) tasks.push(fetchWeatherSearch(q));

  if (intents.includes("marine")) tasks.push(fetchMarineSearch(q));

  if (intents.includes("earthquake")) tasks.push(fetchEarthquakeSearch(q));

  if (intents.includes("currency")) tasks.push(fetchCurrencySearch(q));

  if (intents.includes("distance")) tasks.push(fetchDistanceSearch(q));

  if (intents.includes("places")) tasks.push(fetchPlacesSearch(q));

  if (intents.includes("news")) tasks.push(fetchNewsSearch(q));

  if (intents.includes("aviation")) tasks.push(fetchAviationSearch(q, options?.recentUserText ?? []));

  if (intents.includes("satellite")) tasks.push(fetchIssSearch(q));

  if (intents.includes("spaceweather")) tasks.push(fetchSpaceWeatherSearch(q));

  if (intents.includes("alerts")) tasks.push(fetchIsraelAlertsSearch(q));

  if (intents.includes("disaster")) tasks.push(fetchDisasterSearch(q));

  if (intents.includes("country")) tasks.push(fetchCountrySearch(q));

  if (intents.includes("holiday")) tasks.push(fetchHolidaySearch(q));

  if (intents.includes("government")) tasks.push(fetchGovernmentSearch(q));

  if (intents.includes("github")) tasks.push(fetchGitHubSearch(q));

  if (intents.includes("market")) {
    tasks.push(fetchCoinGeckoSearch(q));
    tasks.push(fetchSearxSearch(q));
  }
  if (intents.includes("reddit")) tasks.push(fetchRedditSearch(q));
  if (intents.includes("hackernews")) tasks.push(fetchHackerNewsSearch(q));
  if (intents.includes("arxiv")) tasks.push(fetchArxivSearch(q));
  if (intents.includes("searx")) tasks.push(fetchSearxSearch(q));

  if (intents.includes("huggingface")) {

    tasks.push(fetchHuggingFaceModelsSearch(q));

    tasks.push(fetchHuggingFaceDatasetsSearch(q));

  }

  if (intents.includes("wikipedia")) {

    const wikiQ = stripSearchVerb(q);

    tasks.push(fetchWikipediaSearch(wikiQ, "en"));

    if (/[\u0590-\u05FF]/.test(q)) {

      tasks.push(fetchWikipediaSearch(wikiQ, "he"));

    }

  }



  const settled = tasks.length ? await Promise.all(tasks) : [];

  const hasLiveData = settled.some((s) => s.ok && s.text.trim());
  const hadStructuredOnly =
    intents.some((i) => STRUCTURED_INTENTS.includes(i)) &&
    !intents.includes("wikipedia") &&
    !intents.includes("searx");

  if (!hasLiveData && hadStructuredOnly) {
    const fallbacks = await runWikiSearxFallback(q);
    settled.push(...fallbacks);
  }

  const okContext = formatWebContext(settled);
  const contextText = okContext.trim() ? okContext : formatWebSearchNoResultsContext();

  return {

    contextText,

    sources: settled,

    summaryHe: summarizeSearchResult(settled, intents),

    intents,

  };

};



/** Back-compat wrapper — returns context string only. */

export const fetchWebContext = async (query: string): Promise<string> => {

  const result = await runWebSearch(query);

  return result.contextText;

};



export { userRequestsSearch } from "./intents";

