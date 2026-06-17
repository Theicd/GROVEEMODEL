// @ts-nocheck
import { fetchRemoteText } from "../fetch/remoteFetch";
import { buildStockSearchQueries, scoreStockCandidate } from "./imageQuery";
import { enqueueStockImageSearch } from "./stockImageQueue";

import { detectStockProvider, type StockImageProvider } from "./imageFields";

export type { StockImageProvider } from "./imageFields";
export { detectStockProvider } from "./imageFields";

export type StockImageResult = {
  url: string;
  provider: StockImageProvider;
  score?: number;
};

export const STOCK_PROVIDER_LABELS: Record<StockImageProvider, string> = {
  openverse: "Openverse",
  wikimedia: "Wikimedia",
  pexels: "Pexels",
  pixabay: "Pixabay",
  unsplash: "Unsplash",
};


type StockCandidate = {
  url: string;
  tags: string;
  title?: string;
  width?: number;
  height?: number;
};

type OpenverseImage = { url?: string; thumbnail?: string; title?: string; tags?: { name?: string }[] };
type OpenverseResponse = { results?: OpenverseImage[] };

type WikimediaResponse = {
  query?: {
    pages?: Record<
      string,
      { title?: string; imageinfo?: { thumburl?: string; url?: string; width?: number; height?: number }[] }
    >;
  };
};

type PexelsPhoto = {
  src?: { large?: string; medium?: string; landscape?: string };
  alt?: string;
  width?: number;
  height?: number;
};
type PexelsResponse = { photos?: PexelsPhoto[] };

type PixabayHit = {
  tags?: string;
  largeImageURL?: string;
  webformatURL?: string;
  imageWidth?: number;
  imageHeight?: number;
};
type PixabayResponse = { hits?: PixabayHit[] };

type UnsplashResponse = {
  results?: { urls?: { regular?: string; small?: string }; alt_description?: string; description?: string }[];
};

const cache = new Map<string, StockImageResult | null>();
const inflight = new Map<string, Promise<StockImageResult | null>>();

function envKey(name: string): string {
  return (import.meta.env[name] as string | undefined)?.trim() ?? "";
}

function pickHttpsUrl(candidates: (string | undefined)[]): string {
  for (const raw of candidates) {
    const url = raw?.trim() ?? "";
    if (!url.startsWith("https://")) continue;
    if (/svg|icon|logo|avatar|badge|1x1|pixel|spacer|blank\.gif/i.test(url)) continue;
    return url;
  }
  return "";
}

function layoutBonus(width?: number, height?: number): number {
  if (!width || !height) return 0;
  if (width < 480 || height < 280) return -6;
  const ratio = width / height;
  if (ratio >= 1.2 && ratio <= 2.2) return 4;
  return 0;
}

function pickBestCandidate(query: string, rows: StockCandidate[]): StockCandidate | null {
  let best: StockCandidate | null = null;
  let bestScore = 0;

  for (const row of rows) {
    const url = pickHttpsUrl([row.url]);
    if (!url) continue;
    const score = scoreStockCandidate(query, row.tags, row.title ?? "") + layoutBonus(row.width, row.height);
    if (score > bestScore) {
      bestScore = score;
      best = { ...row, url };
    }
  }

  if (best && bestScore >= 8) return best;

  for (const row of rows) {
    const url = pickHttpsUrl([row.url]);
    if (url) return { ...row, url };
  }

  return null;
}


async function fetchJson<T>(apiUrl: string, timeoutMs = 14_000): Promise<T | null> {
  try {
    const raw = await fetchRemoteText(apiUrl, timeoutMs);
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function searchOpenverseCandidates(query: string): Promise<StockCandidate[]> {
  const api = `https://api.openverse.org/v1/images/?q=${encodeURIComponent(query)}&page_size=12&license=cc0,by,by-sa&format=json`;
  const parsed = await fetchJson<OpenverseResponse>(api);
  return (parsed?.results ?? [])
    .map((row) => ({
      url: pickHttpsUrl([row.url, row.thumbnail]),
      tags: (row.tags ?? []).map((t) => t.name ?? "").join(" "),
      title: row.title ?? "",
    }))
    .filter((row) => row.url);
}

async function searchWikimediaCandidates(query: string): Promise<StockCandidate[]> {
  const api =
    "https://commons.wikimedia.org/w/api.php?" +
    new URLSearchParams({
      action: "query",
      generator: "search",
      gsrsearch: query,
      gsrlimit: "12",
      gsrnamespace: "6",
      prop: "imageinfo",
      iiprop: "url|size",
      iiurlwidth: "960",
      format: "json",
      origin: "*",
    }).toString();
  const parsed = await fetchJson<WikimediaResponse>(api);
  const pages = parsed?.query?.pages ?? {};
  const out: StockCandidate[] = [];
  for (const page of Object.values(pages)) {
    for (const info of page.imageinfo ?? []) {
      const url = pickHttpsUrl([info.thumburl, info.url]);
      if (!url) continue;
      out.push({
        url,
        tags: page.title ?? "",
        title: page.title ?? "",
        width: info.width,
        height: info.height,
      });
    }
  }
  return out;
}

async function searchPexelsCandidates(query: string): Promise<StockCandidate[]> {
  const api = `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=12&orientation=landscape`;
  const parsed = await fetchJson<PexelsResponse>(api);
  return (parsed?.photos ?? [])
    .map((photo) => ({
      url: pickHttpsUrl([photo.src?.landscape, photo.src?.large, photo.src?.medium]),
      tags: photo.alt ?? "",
      title: photo.alt ?? "",
      width: photo.width,
      height: photo.height,
    }))
    .filter((row) => row.url);
}

async function searchPixabayCandidates(query: string): Promise<StockCandidate[]> {
  const key = envKey("VITE_PIXABAY_API_KEY");
  if (!key) return [];
  const api =
    "https://pixabay.com/api/?" +
    new URLSearchParams({
      key,
      q: query,
      image_type: "photo",
      orientation: "horizontal",
      lang: "en",
      safesearch: "true",
      per_page: "20",
    }).toString();
  const parsed = await fetchJson<PixabayResponse>(api);
  return (parsed?.hits ?? [])
    .map((hit) => ({
      url: pickHttpsUrl([hit.largeImageURL, hit.webformatURL]),
      tags: hit.tags ?? "",
      title: hit.tags ?? "",
      width: hit.imageWidth,
      height: hit.imageHeight,
    }))
    .filter((row) => row.url);
}

async function searchUnsplashCandidates(query: string): Promise<StockCandidate[]> {
  const key = envKey("VITE_UNSPLASH_ACCESS_KEY");
  if (!key) return [];
  const api = `https://api.unsplash.com/search/photos?query=${encodeURIComponent(query)}&per_page=12&orientation=landscape&client_id=${encodeURIComponent(key)}`;
  const parsed = await fetchJson<UnsplashResponse>(api);
  return (parsed?.results ?? [])
    .map((row) => ({
      url: pickHttpsUrl([row.urls?.regular, row.urls?.small]),
      tags: `${row.alt_description ?? ""} ${row.description ?? ""}`.trim(),
      title: row.alt_description ?? row.description ?? "",
    }))
    .filter((row) => row.url);
}

type ProviderSearch = {
  provider: StockImageProvider;
  search: (query: string) => Promise<StockCandidate[]>;
  enabled: () => boolean;
};

/** Pixabay first when keyed — same strategy as PIXEL-ISR (with relevance scoring). */
const PROVIDERS: ProviderSearch[] = [
  {
    provider: "pixabay",
    search: searchPixabayCandidates,
    enabled: () => Boolean(envKey("VITE_PIXABAY_API_KEY")),
  },
  { provider: "pexels", search: searchPexelsCandidates, enabled: () => true },
  { provider: "openverse", search: searchOpenverseCandidates, enabled: () => true },
  { provider: "wikimedia", search: searchWikimediaCandidates, enabled: () => true },
  {
    provider: "unsplash",
    search: searchUnsplashCandidates,
    enabled: () => Boolean(envKey("VITE_UNSPLASH_ACCESS_KEY")),
  },
];

async function searchProviders(queries: string[]): Promise<StockImageResult | null> {
  let best: StockImageResult | null = null;

  for (const query of queries) {
    for (const row of PROVIDERS) {
      if (!row.enabled()) continue;
      const candidates = await row.search(query);
      const picked = pickBestCandidate(query, candidates);
      if (!picked) continue;

      const score = scoreStockCandidate(query, picked.tags, picked.title ?? "") + layoutBonus(picked.width, picked.height);
      const hit: StockImageResult = { url: picked.url, provider: row.provider, score };

      if (!best || (hit.score ?? 0) > (best.score ?? 0)) best = hit;
      if ((hit.score ?? 0) >= 18) return hit;
    }
  }

  return best;
}

/** Search free stock libraries and return the best relevance match. */
export async function searchStockImage(title: string, hint = ""): Promise<StockImageResult | null> {
  const queries = buildStockSearchQueries(title, hint);
  if (!queries.length) return null;

  const key = queries.join("|").toLowerCase();
  if (cache.has(key)) return cache.get(key)!;

  const pending = inflight.get(key);
  if (pending) return pending;

  const job = enqueueStockImageSearch(async () => {
    try {
      const hit = await searchProviders(queries);
      cache.set(key, hit);
      return hit;
    } catch {
      cache.set(key, null);
      return null;
    } finally {
      inflight.delete(key);
    }
  });

  inflight.set(key, job);
  return job;
}

/** Back-compat helper — URL only. */
export async function searchStockImageUrl(title: string, hint = ""): Promise<string> {
  const hit = await searchStockImage(title, hint);
  return hit?.url ?? "";
}

export function listConfiguredStockProviders(): StockImageProvider[] {
  return PROVIDERS.filter((p) => p.enabled()).map((p) => p.provider);
}

export function clearStockImageCacheForTests(): void {
  cache.clear();
  inflight.clear();
}

// Re-export for tests / callers
export { buildImageSearchQuery } from "./imageQuery";
