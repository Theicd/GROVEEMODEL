/** Web search result types — shared by providers, orchestrator, and UI. */

export type SearchProviderId =
  | "open-meteo"
  | "open-meteo-marine"
  | "usgs-earthquake"
  | "wikipedia-en"
  | "wikipedia-he"
  | "github"
  | "huggingface-models"
  | "huggingface-datasets"
  | "world-time"
  | "rest-countries"
  | "nager-holidays"
  | "wikidata-gov"
  | "frankfurter-fx"
  | "osrm-distance"
  | "nominatim-places"
  | "news-rss"
  | "adsb-aviation"
  | "iss-tracker"
  | "noaa-space"
  | "israel-alerts"
  | "gdacs-disasters"
  | "searxng"
  | "reddit"
  | "hackernews"
  | "arxiv"
  | "coingecko";

export type SearchSourceResult = {
  provider: SearchProviderId;
  label: string;
  ok: boolean;
  text: string;
  url?: string;
  error?: string;
  latencyMs: number;
};

export type WebSearchResult = {
  /** Flat text injected into the LLM system prompt. */
  contextText: string;
  sources: SearchSourceResult[];
  /** Short Hebrew summary for status / UI header. */
  summaryHe: string;
  intents: SearchIntent[];
};

export type SearchIntent =
  | "weather"
  | "marine"
  | "earthquake"
  | "github"
  | "huggingface"
  | "wikipedia"
  | "worldtime"
  | "country"
  | "holiday"
  | "government"
  | "currency"
  | "distance"
  | "places"
  | "news"
  | "aviation"
  | "satellite"
  | "spaceweather"
  | "alerts"
  | "disaster"
  | "searx"
  | "reddit"
  | "hackernews"
  | "arxiv"
  | "market";

export type FetchJsonOptions = {
  timeoutMs?: number;
  headers?: Record<string, string>;
};

export type WebSearchOptions = {
  /** Recent user messages — used for aviation region / follow-up context. */
  recentUserText?: string[];
};
