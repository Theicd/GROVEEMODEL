/** Web search result types — shared by providers, orchestrator, and UI. */

import type { TimeWidgetData } from "../timeWidget/types";
import type { WeatherWidgetData } from "../weatherWidget/types";

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
  | "openserp"
  | "tavily"
  | "scavio"
  | "open-meteo-air-quality"
  | "arxiv"
  | "url-context"
  | "movie-catalog"
  | "pixabay-images"
  | "pixabay-videos"
  | "peertube-videos"
  | "internet-archive-media"
  | "invidious-videos"
  | "israeli-products"
  | "live-tv";

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
  | "link"
  | "movies"
  | "images"
  | "video"
  | "products"
  | "livemedia";

export type DataTier = "structured" | "news" | "web_fallback";

export type AnswerShape = "short_fact" | "bullet_list" | "overview" | "count";

export type SearchBriefLink = { label: string; url: string };

/** Movie / TV metadata row from Wikidata, TVMaze, Archive.org, or TMDB. */
export type MovieSerpHit = {
  id: string;
  title: string;
  titleHe?: string;
  originalTitle?: string;
  year?: number;
  url: string;
  snippet: string;
  poster?: string;
  runtime?: number;
  genres?: string[];
  ageRating?: string;
  seeds?: number;
  quality?: string;
  source?: string;
  rating?: number;
  /** Direct mp4/webm URL when Internet Archive item is playable in-browser. */
  playUrl?: string;
  durationSec?: number;
};

/** Stock image / video row from Pixabay (PIXEL-ISR). */
export type MediaSerpHit = {
  id: string;
  mediaType: "image" | "video";
  title: string;
  url: string;
  playUrl: string;
  downloadUrl?: string;
  thumbnail: string;
  snippet?: string;
  author?: string;
  licenseUrl?: string;
  tags?: string;
  durationSec?: number;
  width?: number;
  height?: number;
  source?: string;
  /** YouTube-specific result shape when source is YouTube / Invidious. */
  youtubeSubType?: "video" | "playlist" | "channel";
};

/** Live TV channel or internet radio station from IPTV-org / Radio Browser. */
export type LiveMediaSerpHit = {
  id: string;
  mediaType: "livetv" | "radio";
  title: string;
  url: string;
  streamUrl: string;
  logoUrl?: string;
  snippet?: string;
  country?: string;
  category?: string;
  tags?: string[];
  status?: "working" | "warning" | "offline" | "unknown";
  bitrate?: number;
  codec?: string;
  votes?: number;
  fuseScore?: number;
};

/** Israeli supermarket product row (catalog + Open Food Facts + optional Cheapersal prices). */
export type ProductSerpHit = {
  id: string;
  barcode: string;
  title: string;
  brand?: string;
  category?: string;
  url: string;
  snippet: string;
  imageUrl?: string;
  source: string;
  /** Cheapest price in NIS (Cheapersal). */
  priceNis?: number;
  priceMaxNis?: number;
  priceAvgNis?: number;
  cheapestChain?: string;
  priceStoreCount?: number;
  unitQty?: string;
  priceSummary?: string;
};

/** Single web search result row (SearXNG / unified SERP). */
export type WebSerpHit = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  engine?: string;
};

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
  weatherWidget?: WeatherWidgetData;
  /** RSS cards attached by grovee-news provider. */
  newsCards?: import("../groveeNews/types").GroveeNewsCard[];
  /** Human-readable RSS scan status for search UI. */
  newsScanNote?: string;
  /** Structured web hits from SearXNG. */
  webHits?: WebSerpHit[];
  /** Enriched movie rows from movie catalog providers. */
  movieHits?: MovieSerpHit[];
  /** Stock photos / videos from Pixabay. */
  mediaHits?: MediaSerpHit[];
  /** Israeli supermarket products (barcode catalog + OFF). */
  productHits?: ProductSerpHit[];
  /** Hugging Face models with API probe / connection snippets. */
  hfModelHits?: import("./hf/hfModelTypes").HfModelSerpHit[];
  /** IPTV / Radio Browser hits from local live media library. */
  liveMediaHits?: LiveMediaSerpHit[];
  /** Coordinates / route geometry for map panel (Nominatim, OSRM). */
  geo?: {
    lat?: number;
    lon?: number;
    label?: string;
    from?: { lat: number; lon: number; label?: string };
    to?: { lat: number; lon: number; label?: string };
    route?: Array<{ lat: number; lon: number }>;
  };
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
  /** Side-panel SERP — force broad web + news + wiki/github enrichment. */
  panelSearch?: boolean;
};
