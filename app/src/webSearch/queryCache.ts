import type { SearchProviderId, SearchSourceResult } from "./types";

type CacheEntry = {
  result: SearchSourceResult;
  expiresAt: number;
};

const cache = new Map<string, CacheEntry>();

/** TTL per provider — shorter for news, longer for static-ish data. */
const TTL_MS: Partial<Record<SearchProviderId, number>> = {
  "open-meteo": 5 * 60_000,
  "open-meteo-air-quality": 5 * 60_000,
  "open-meteo-marine": 5 * 60_000,
  "news-rss": 3 * 60_000,
  "frankfurter-fx": 30 * 60_000,
  "yahoo-finance": 15 * 60_000,
  coingecko: 2 * 60_000,
  "usgs-earthquake": 5 * 60_000,
  "adsb-aviation": 90_000,
  "iss-tracker": 90_000,
  github: 10 * 60_000,
  "huggingface-models": 10 * 60_000,
  "huggingface-datasets": 10 * 60_000,
  "nominatim-places": 10 * 60_000,
  "osm-overpass-marine": 10 * 60_000,
  "ais-ships": 90_000,
  "wikipedia-en": 30 * 60_000,
  "wikipedia-he": 30 * 60_000,
  searxng: 5 * 60_000,
  arxiv: 15 * 60_000,
  "url-context": 10 * 60_000,
};

const DEFAULT_TTL_MS = 5 * 60_000;

export const cacheKey = (provider: SearchProviderId, query: string): string =>
  `${provider}:${query.trim().toLowerCase().replace(/\s+/g, " ")}`;

export const getCachedSearchResult = (
  provider: SearchProviderId,
  query: string,
): SearchSourceResult | null => {
  const key = cacheKey(provider, query);
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  return { ...entry.result, latencyMs: 0 };
};

export const setCachedSearchResult = (
  provider: SearchProviderId,
  query: string,
  result: SearchSourceResult,
): void => {
  if (!result.ok) return;
  const ttl = TTL_MS[provider] ?? DEFAULT_TTL_MS;
  cache.set(cacheKey(provider, query), {
    result,
    expiresAt: Date.now() + ttl,
  });
};

/** Wrap a provider fetch with in-memory TTL cache. */
export const wrapWithQueryCache = async (
  provider: SearchProviderId,
  query: string,
  fetch: () => Promise<SearchSourceResult>,
): Promise<SearchSourceResult> => {
  const cached = getCachedSearchResult(provider, query);
  if (cached) return cached;
  const result = await fetch();
  setCachedSearchResult(provider, query, result);
  return result;
};

export const clearQueryCache = (): void => {
  cache.clear();
};

export const queryCacheSize = (): number => cache.size;
