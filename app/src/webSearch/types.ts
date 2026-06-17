/** Web search result types — shared by providers, orchestrator, and UI. */

import type { TimeWidgetData } from "../timeWidget/types";

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
  | "grovee-news"
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
  | "spacex-launches"
  | "searxng"
  | "open-meteo-air-quality"
  | "arxiv"
  | "url-context";

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
  | "spacex"
  | "airquality"
  | "arxiv"
  | "link";

export type DataTier = "structured" | "news" | "web_fallback";

export type AnswerShape = "short_fact" | "bullet_list" | "overview" | "count";

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
  timeWidget?: TimeWidgetData;
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

export type SharedSearchRegion = {
  label: string;
  place: {
    name: string;
    latitude: number;
    longitude: number;
    elevation?: number;
    country_code?: string;
    admin1?: string;
    timezone?: string;
  };
  phrase: string;
};

export type WebSearchPlanHint = {
  intents?: SearchIntent[];
  queries?: string[];
  answerShape?: AnswerShape;
  useWebFallback?: boolean;
  blendNewsWithWeb?: boolean;
};

export type WebSearchOptions = {
  /** Recent user messages — used for aviation region / follow-up context. */
  recentUserText?: string[];
  onProgress?: (event: SearchProgressEvent) => void;
  /** Optional planner output — merges with regex routing. */
  plan?: WebSearchPlanHint;
  /** Single geocode for cross-source / multi-geo intents (Phase 4). */
  sharedRegion?: SharedSearchRegion;
};
