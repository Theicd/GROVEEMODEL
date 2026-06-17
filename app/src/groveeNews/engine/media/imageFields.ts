// @ts-nocheck
export type StockImageProvider = "openverse" | "wikimedia" | "pexels" | "pixabay" | "unsplash";

const STOCK_HOST_HINTS: { host: RegExp; provider: StockImageProvider }[] = [
  { host: /(^|\.)pexels\.com$/i, provider: "pexels" },
  { host: /(^|\.)pixabay\.com$/i, provider: "pixabay" },
  { host: /(^|\.)unsplash\.com$/i, provider: "unsplash" },
  { host: /(^|\.)wikimedia\.org$/i, provider: "wikimedia" },
  { host: /(^|\.)flickr\.com$/i, provider: "openverse" },
  { host: /(^|\.)staticflickr\.com$/i, provider: "openverse" },
  { host: /(^|\.)openverse\.org$/i, provider: "openverse" },
];

/** Coerce RSS/DB image fields to a safe URL string. */
export function normalizeImageUrl(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "";
}

export function hasImageUrl(value: unknown): boolean {
  return normalizeImageUrl(value).length > 0;
}

export function detectStockProvider(url: string): StockImageProvider | null {
  try {
    const host = new URL(url).hostname;
    for (const row of STOCK_HOST_HINTS) {
      if (row.host.test(host)) return row.provider;
    }
  } catch {
    /* ignore */
  }
  return null;
}

export function isStockImageUrl(value: unknown): boolean {
  const url = normalizeImageUrl(value);
  return url ? detectStockProvider(url) !== null : false;
}

/** True for RSS / article / page images — not a stock-library placeholder. */
export function hasRealImageUrl(value: unknown): boolean {
  return hasImageUrl(value) && !isStockImageUrl(value);
}
