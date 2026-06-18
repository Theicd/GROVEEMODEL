import { fetchJson } from "../fetchJson";
import { proxyAwareFetch } from "../proxyFetch";
import { ADDITLIST_IMAGE_CHAINS, cheapersalCatalogImage } from "./israeliProductLinks";
import { fetchCheapersalPageMeta, fetchCheapersalPublicMeta, isCheapersalProductUrl } from "./cheapersalPublicPage";

export const isGenericCatalogImage = (url?: string): boolean =>
  Boolean(url?.includes("additlist.com/images/catalog/"));

/** OFF image path from EAN-13 barcode. */
export const openFoodFactsImageCandidates = (barcode: string): string[] => {
  const code = barcode.trim();
  if (!/^\d{8,14}$/.test(code)) return [];
  const ean = code.length === 13 ? code : code.padStart(13, "0");
  const path = `${ean.slice(0, 3)}/${ean.slice(3, 6)}/${ean.slice(6, 9)}/${ean.slice(9)}`;
  return [
    `https://images.openfoodfacts.org/images/products/${path}/front_he.400.jpg`,
    `https://images.openfoodfacts.org/images/products/${path}/front_en.400.jpg`,
    `https://images.openfoodfacts.org/images/products/${path}/front.400.jpg`,
    `https://images.openfoodfacts.org/images/products/${path}/front_he.jpg`,
  ];
};

export async function probeImageUrl(url: string): Promise<boolean> {
  const target = url.trim();
  if (!target) return false;
  try {
    const res = await proxyAwareFetch(target, {
      method: "GET",
      headers: { Range: "bytes=0-256" },
    });
    return res.ok || res.status === 206;
  } catch {
    return false;
  }
}

async function lookupOpenFoodFactsImage(barcode: string): Promise<string | undefined> {
  try {
    const data = await fetchJson<{
      status?: number;
      product?: { image_front_small_url?: string; image_front_url?: string };
    }>(
      `https://world.openfoodfacts.org/api/v2/product/${encodeURIComponent(barcode)}.json?fields=image_front_url,image_front_small_url`,
      undefined,
      { timeoutMs: 8000 },
    );
    if (data.status !== 1) return undefined;
    const candidates = [data.product?.image_front_small_url, data.product?.image_front_url].filter(
      Boolean,
    ) as string[];
    for (const url of candidates) {
      if (await probeImageUrl(url)) return url;
    }
  } catch {
    /* ignore */
  }
  return undefined;
}

async function probeAdditlistChains(barcode: string): Promise<string | undefined> {
  for (const chain of ADDITLIST_IMAGE_CHAINS) {
    const url = `https://price-api.additlist.com/images/catalog/${chain}/${barcode.trim()}.jpg`;
    if (await probeImageUrl(url)) return url;
  }
  return undefined;
}

/**
 * Resolve a working product image URL (never returns a known-broken catalog guess).
 */
export async function resolveProductImageUrl(
  barcode: string,
  preferred?: string,
  pageUrl?: string,
): Promise<string | undefined> {
  const code = barcode.trim();
  if (!/^\d{8,14}$/.test(code)) return undefined;

  if (preferred?.trim() && !isGenericCatalogImage(preferred)) {
    if (await probeImageUrl(preferred)) return preferred.trim();
  }

  const pub =
    (pageUrl && isCheapersalProductUrl(pageUrl) ? await fetchCheapersalPageMeta(pageUrl) : null) ??
    (await fetchCheapersalPublicMeta(code));
  if (pub?.imageUrl && (await probeImageUrl(pub.imageUrl))) return pub.imageUrl;

  const offApi = await lookupOpenFoodFactsImage(code);
  if (offApi) return offApi;

  const additlist = await probeAdditlistChains(code);
  if (additlist) return additlist;

  for (const url of openFoodFactsImageCandidates(code)) {
    if (await probeImageUrl(url)) return url;
  }

  const rami = `https://static.rfrsh.co.il/supermarket/product/${code}/small.jpg`;
  if (await probeImageUrl(rami)) return rami;

  if (preferred?.trim() && !isGenericCatalogImage(preferred)) return preferred.trim();
  return undefined;
}

export const productHasDisplayableImage = (hit: { imageUrl?: string }): boolean =>
  Boolean(hit.imageUrl?.trim() && !isGenericCatalogImage(hit.imageUrl));

export const sortProductHitsByRichness = <T extends { priceNis?: number; imageUrl?: string }>(
  hits: T[],
): T[] =>
  [...hits].sort((a, b) => {
    const score = (h: T) =>
      (h.priceNis != null ? 4 : 0) + (productHasDisplayableImage(h) ? 2 : 0);
    return score(b) - score(a);
  });

/** Strip placeholder catalog URLs that often 404 (bread, etc.). */
export const stripGenericCatalogImage = (imageUrl?: string): string | undefined => {
  if (!imageUrl?.trim()) return undefined;
  return isGenericCatalogImage(imageUrl) ? undefined : imageUrl.trim();
};

export const defaultCatalogImageGuess = (barcode: string): string => cheapersalCatalogImage(barcode);
