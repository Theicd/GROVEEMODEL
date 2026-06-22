import type { SearchProviderId, SearchSourceResult } from "./types";
import { agentDebugLog } from "../debugAgentLog";

type CacheEntry = {
  result: SearchSourceResult;
  expiresAt: number;
};

const cache = new Map<string, CacheEntry>();

/**
 * Live providers — no cross-turn cache (user expects fresh fetches every question).
 * TTL applies only within the same runWebSearch when the same provider+query is hit twice.
 */
const LIVE_PROVIDER_IDS = new Set<SearchProviderId>([
  "open-meteo",
  "open-meteo-air-quality",
  "open-meteo-marine",
  "grovee-news",
  "usgs-earthquake",
  "adsb-aviation",
  "iss-tracker",
  "ais-ships",
  "coingecko",
  "yahoo-finance",
  "searxng",
  "openserp",
  "tavily",
  "scavio",
  "gdacs-disasters",
  "israel-alerts",
  "noaa-space",
  "spacex-launches",
  "celestrak",
  "starlink-catalog",
]);

/** TTL per provider — static-ish only; live providers use LIVE_CACHE_TTL_MS. */
const TTL_MS: Partial<Record<SearchProviderId, number>> = {
  "frankfurter-fx": 30 * 60_000,
  github: 10 * 60_000,
  "huggingface-models": 10 * 60_000,
  "huggingface-datasets": 10 * 60_000,
  "nominatim-places": 10 * 60_000,
  "osm-overpass-marine": 10 * 60_000,
  "wikipedia-en": 30 * 60_000,
  "wikipedia-he": 30 * 60_000,
  arxiv: 15 * 60_000,
  "url-context": 10 * 60_000,
};

const DEFAULT_TTL_MS = 5 * 60_000;
const LIVE_CACHE_TTL_MS = 0;

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
  const ttl = LIVE_PROVIDER_IDS.has(provider)
    ? LIVE_CACHE_TTL_MS
    : (TTL_MS[provider] ?? DEFAULT_TTL_MS);
  if (ttl <= 0) return;
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
  if (cached) {
    // #region agent log
    agentDebugLog("H6", "queryCache.ts:wrapWithQueryCache", "cache hit", {
      provider,
      queryPreview: query.slice(0, 80),
      ok: cached.ok,
      textLen: cached.text.length,
    });
    // #endregion
    return cached;
  }
  const result = await fetch();
  setCachedSearchResult(provider, query, result);
  return result;
};

export const clearQueryCache = (): void => {
  cache.clear();
};

export const queryCacheSize = (): number => cache.size;
