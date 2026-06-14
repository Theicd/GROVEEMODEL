import type { SearchSourceResult } from "../webSearch/types";

const STORAGE_KEY = "grovee-starlink-catalog-v1";
export const STARLINK_CATALOG_MAX_AGE_MS = 30 * 60 * 1000;

export type StarlinkCatalogCache = {
  fetchedAt: number;
  total: number;
  sample: string[];
};

let memoryCache: StarlinkCatalogCache | null = null;

export const loadStarlinkCatalogCache = (): StarlinkCatalogCache | null => {
  if (memoryCache && Date.now() - memoryCache.fetchedAt < STARLINK_CATALOG_MAX_AGE_MS) {
    return memoryCache;
  }
  if (typeof localStorage === "undefined") return memoryCache;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return memoryCache;
    const parsed = JSON.parse(raw) as StarlinkCatalogCache;
    if (!parsed?.total || !parsed.fetchedAt) return memoryCache;
    memoryCache = parsed;
    return parsed;
  } catch {
    return memoryCache;
  }
};

export const saveStarlinkCatalogCache = (cache: StarlinkCatalogCache): void => {
  memoryCache = cache;
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cache));
  } catch {
    /* quota */
  }
};

export const getFreshStarlinkCatalogCache = (maxAgeMs = STARLINK_CATALOG_MAX_AGE_MS): StarlinkCatalogCache | null => {
  const c = loadStarlinkCatalogCache();
  if (!c) return null;
  if (Date.now() - c.fetchedAt > maxAgeMs) return null;
  return c;
};

export const formatStarlinkCatalogText = (cache: StarlinkCatalogCache, stale = false): string => {
  const when = new Date(cache.fetchedAt).toISOString().replace("T", " ").slice(0, 19);
  return [
    `ANSWER (Starlink active): ${cache.total}`,
    `לווייני Starlink בקטalog CelesTrak (GROUP=starlink): ${cache.total}`,
    `עודכן: ${when} UTC${stale ? " (cache)" : ""}`,
    "מקור: CelesTrak — אותו מאגר TLE שמוגדר בעולם חי (api-registry starlink).",
    ...(cache.sample.length ? ["דוגמאות:", ...cache.sample] : []),
  ].join("\n");
};

export const starlinkSearchResultFromCache = (_query: string, maxAgeMs?: number): SearchSourceResult | null => {
  const cache = getFreshStarlinkCatalogCache(maxAgeMs ?? STARLINK_CATALOG_MAX_AGE_MS);
  if (!cache) return null;
  return {
    provider: "starlink-catalog",
    label: "Starlink (CelesTrak / עולם חי)",
    ok: true,
    text: formatStarlinkCatalogText(cache, true),
    url: "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    latencyMs: 0,
  };
};

/** Parse first N satellite names from CelesTrak TLE text. */
export const sampleNamesFromTle = (text: string, n = 6): string[] => {
  const lines = text.split("\n");
  const out: string[] = [];
  for (let i = 0; i + 2 < lines.length && out.length < n; i += 3) {
    const name = lines[i]?.trim();
    const tle1 = lines[i + 1]?.trim() ?? "";
    if (!name || !tle1.startsWith("1 ")) continue;
    const norad = parseInt(tle1.substring(2, 7), 10);
    out.push(`${out.length + 1}. ${name} (NORAD ${Number.isFinite(norad) ? norad : "?"})`);
  }
  return out;
};

export const countTleSatellites = (text: string): number => {
  let count = 0;
  for (const line of text.split("\n")) {
    if (line.startsWith("1 ")) count++;
  }
  return count;
};

export const clearStarlinkCatalogCacheForTests = (): void => {
  memoryCache = null;
  if (typeof localStorage !== "undefined") {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }
};
