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
  | "coingecko"
  | "stooq-commodity"
  | "yahoo-finance"
  | "hacker-news"
  | "market-stocks"
  | "reddit"
  | "flight-status"
  | "youtube"
  | "ais-ships"
  | "osm-overpass-marine"
  | "celestrak"
  | "starlink-catalog"
  | "spacex-launches";

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
  | "crypto"
  | "commodity"
  | "market"
  | "hackernews"
  | "youtube"
  | "ships"
  | "marine-infra"
  | "spacex";

export type SearchBriefLink = { label: string; url: string };

export type SearchBrief = {
  facts: string[];
  links: SearchBriefLink[];
  gaps: string[];
  intents: SearchIntent[];
};

export type SearchSourceResult = {
  provider: SearchProviderId;
  label: string;
  ok: boolean;
  text: string;
  url?: string;
  error?: string;
  latencyMs: number;
};

export type SearchProgressEvent =
  | { type: "start"; intents: SearchIntent[]; query: string }
  | { type: "provider_done"; result: SearchSourceResult }
  | { type: "complete"; sources: SearchSourceResult[] };

export type WebSearchResult = {
  /** Compact brief injected into the LLM system prompt. */
  contextText: string;
  sources: SearchSourceResult[];
  /** Short Hebrew summary for status / UI header. */
  summaryHe: string;
  intents: SearchIntent[];
  brief?: SearchBrief;
  /** Fixed Hebrew reply from live providers — bypasses LLM when set. */
  cannedReply?: string | null;
};

export type FetchJsonOptions = {
  timeoutMs?: number;
  headers?: Record<string, string>;
};

export type WebSearchOptions = {
  /** Recent user messages — used for aviation region / follow-up context. */
  recentUserText?: string[];
  onProgress?: (event: SearchProgressEvent) => void;
};
