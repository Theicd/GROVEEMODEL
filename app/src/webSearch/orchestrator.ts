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

import type { SearchSourceResult, WebSearchResult } from "./types";



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



/** Run routed parallel search — typically 1–5 s total. */

export const runWebSearch = async (query: string): Promise<WebSearchResult> => {

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

  if (intents.includes("aviation")) tasks.push(fetchAviationSearch(q));

  if (intents.includes("satellite")) tasks.push(fetchIssSearch(q));

  if (intents.includes("spaceweather")) tasks.push(fetchSpaceWeatherSearch(q));

  if (intents.includes("alerts")) tasks.push(fetchIsraelAlertsSearch(q));

  if (intents.includes("disaster")) tasks.push(fetchDisasterSearch(q));

  if (intents.includes("country")) tasks.push(fetchCountrySearch(q));

  if (intents.includes("holiday")) tasks.push(fetchHolidaySearch(q));

  if (intents.includes("government")) tasks.push(fetchGovernmentSearch(q));

  if (intents.includes("github")) tasks.push(fetchGitHubSearch(q));

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



  const settled = await Promise.all(tasks);

  const contextText = formatWebContext(settled);

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

