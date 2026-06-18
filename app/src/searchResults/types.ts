import type { SearchProviderId } from "../webSearch/types";

export type SearchHitKind =
  | "rss"
  | "web"
  | "github"
  | "arxiv"
  | "hackernews"
  | "structured"
  | "movie"
  | "image"
  | "video"
  | "youtube"
  | "product"
  | "hfmodel";

export type SearchResultsFilter =
  | "all"
  | "rss"
  | "web"
  | "images"
  | "video"
  | "youtube"
  | "movies"
  | "products"
  | "hfmodels"
  | "repos";

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
    hfStatus?: string;
    hfProvider?: string;
    hfAccess?: string;
    hfLatency?: number;
    hfCurl?: string;
    hfPython?: string;
    hfCategory?: string;
    hfPipeline?: string;
    hfProbeSource?: string;
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
  repos: number;
  papers: number;
  movies: number;
  images: number;
  videos: number;
  youtube: number;
  products: number;
  hfModels: number;
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
  /** When true, panel opens on «סרטים» filter if movie hits exist. */
  preferMoviesFilter?: boolean;
  preferImagesFilter?: boolean;
  preferVideoFilter?: boolean;
  preferYouTubeFilter?: boolean;
  preferProductsFilter?: boolean;
  /** When true, panel prioritizes Hugging Face model cards. */
  preferHfModelsFilter?: boolean;
};
