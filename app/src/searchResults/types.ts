import type { SearchProviderId } from "../webSearch/types";

export type SearchHitKind =
  | "rss"
  | "web"
  | "github"
  | "arxiv"
  | "hackernews"
  | "structured"
  | "earthquake"
  | "disaster"
  | "ship"
  | "weather"
  | "marine"
  | "place"
  | "route"
  | "movie"
  | "image"
  | "video"
  | "youtube"
  | "livetv"
  | "radio"
  | "product"
  | "hfmodel";

export type SearchResultsFilter =
  | "all"
  | "rss"
  | "web"
  | "images"
  | "video"
  | "youtube"
  | "livetv"
  | "radio"
  | "movies"
  | "products"
  | "hfmodels"
  | "repos"
  | "earthquakes"
  | "disasters"
  | "ships"
  | "events"
  | "weather"
  | "places"
  | "companion";

export type UnifiedSearchHit = {
  id: string;
  kind: SearchHitKind;
  title: string;
  url: string;
  snippet: string;
  /** Source text before UI translation. */
  titleOriginal?: string;
  snippetOriginal?: string;
  imageUrl?: string;
  sourceLabel: string;
  sourceKey?: string;
  faviconUrl?: string;
  provider: SearchProviderId;
  publishedTs?: number;
  score?: number;
  meta?: { stars?: number; engine?: string; year?: number; priceNis?: number;
    magnitude?: number;
    alertLevel?: string;
    disasterType?: string;
    status?: string;
    loadTimeMs?: number;
    hfStatus?: string;
    hfProvider?: string;
    hfAccess?: string;
    hfLatency?: number;
    hfCurl?: string;
    hfPython?: string;
    hfCategory?: string;
    hfPipeline?: string;
    hfProbeSource?: string;
    shipLat?: number;
    shipLon?: number;
    speedKn?: number;
    destination?: string;
    shipSource?: "ais" | "globe" | "route-marker" | "aisstream";
    marineInfraKind?: string;
    regionLabel?: string;
  };
  mediaPlayUrl?: string;
  /** When true, lightbox plays via iframe embed (Invidious / PeerTube page) instead of direct file. */
  mediaEmbedMode?: boolean;
  downloadUrl?: string;
  durationSec?: number;
  author?: string;
  summarizable: boolean;
};

export type SearchResultsFacets = {
  rss: number;
  web: number;
  /** OpenSERP / Grove Search Companion SERP hits. */
  companionWeb: number;
  repos: number;
  papers: number;
  movies: number;
  images: number;
  videos: number;
  youtube: number;
  liveTv: number;
  radio: number;
  products: number;
  hfModels: number;
  earthquakes: number;
  disasters: number;
  ships: number;
  weather: number;
  marine: number;
  places: number;
  other: number;
};

export type SearchResultsPayload = {
  query: string;
  generatedAt: number;
  hits: UnifiedSearchHit[];
  facets: SearchResultsFacets;
  providerErrors: string[];
  /** When true, panel opens on «חדשות RSS» filter if RSS hits exist. */
  preferRssFilter?: boolean;
  /** When true, panel opens on «אתרים» for open-web topic searches. */
  preferWebFilter?: boolean;
  /** When true, panel opens on «חיפוש מקומי» (OpenSERP plugin). */
  preferCompanionFilter?: boolean;
  /** Show dedicated OpenSERP tab (attempted or has hits). */
  showCompanionTab?: boolean;
  /** Last OpenSERP error for empty-state messaging. */
  companionWebError?: string;
  /** When true, panel opens on «סרטים» filter if movie hits exist. */
  preferMoviesFilter?: boolean;
  preferImagesFilter?: boolean;
  preferVideoFilter?: boolean;
  preferYouTubeFilter?: boolean;
  preferLiveTvFilter?: boolean;
  preferRadioFilter?: boolean;
  preferProductsFilter?: boolean;
  /** When true, panel prioritizes Hugging Face model cards. */
  preferHfModelsFilter?: boolean;
  /** When true, panel opens on «רעידות אדמה» filter. */
  preferEarthquakesFilter?: boolean;
  /** When true, panel opens on «אסונות» filter. */
  preferDisastersFilter?: boolean;
  /** When true, panel opens on «אוניות / ים» filter. */
  preferShipsFilter?: boolean;
  /** When true, panel opens on «אירועים / חריגות» (earthquakes + disasters). */
  preferEventsFilter?: boolean;
  /** When true, panel opens on «מזג אוויר / ים». */
  preferWeatherFilter?: boolean;
  /** When true, panel opens on «מקומות / מסלולים». */
  preferPlacesFilter?: boolean;
  /** Live USGS/GDACS status line. */
  liveDisastersNote?: string;
  /** Live AIS / marine infra status line. */
  liveShipsNote?: string;
  /** RSS scan status line from GROVEE NEWS provider. */
  newsRssNote?: string;
};
