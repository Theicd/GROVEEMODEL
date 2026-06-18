import { fetchJson } from "../fetchJson";
import type { ProductSerpHit } from "../types";
import { supermarketProductUrl } from "./israeliProductLinks";

import {
  isGenericCatalogImage,
  resolveProductImageUrl,
  sortProductHitsByRichness,
  stripGenericCatalogImage,
} from "./productImageResolve";

import { applyPublicMetaToHit, fetchCheapersalPageMeta, fetchCheapersalPublicMeta, isCheapersalProductUrl } from "./cheapersalPublicPage";
import { cheapersalProductUrl } from "./israeliProductLinks";

type CheapersalChain = { id?: number; name?: string };
type CheapersalPriceRow = {
  price?: number;
  chain?: CheapersalChain;
  promo?: { promoPrice?: number };
};
type CheapersalPricesPayload = {
  success?: boolean;
  data?: {
    product?: { barcode?: string; name?: string; unitQty?: string; image?: string; imageUrl?: string };
    prices?: CheapersalPriceRow[];
    summary?: {
      cheapest?: number;
      mostExpensive?: number;
      average?: number;
      storeCount?: number;
      cheapestChain?: CheapersalChain;
    };
  };
  error?: string;
};

const CHEAPERSAL_UPSTREAM = "https://api.cheapersal.co.il/api/v1";

export const cheapersalBaseUrl = (): string => {
  const custom = import.meta.env.VITE_CHEAPERSAL_URL?.trim();
  if (custom) return custom.replace(/\/$/, "");
  if (import.meta.env.DEV) return "/api/cheapersal";
  return CHEAPERSAL_UPSTREAM;
};

/** True when API key, custom URL, or dev proxy may reach Cheapersal API. */
export const isCheapersalConfigured = (): boolean =>
  Boolean(import.meta.env.VITE_CHEAPERSAL_API_KEY?.trim()) ||
  import.meta.env.DEV ||
  Boolean(import.meta.env.VITE_CHEAPERSAL_URL?.trim());

const cheapersalHeaders = (): Record<string, string> => {
  const key = import.meta.env.VITE_CHEAPERSAL_API_KEY?.trim();
  if (key) return { "X-API-Key": key };
  return {};
};

export const formatProductPriceSummary = (hit: ProductSerpHit): string => {
  if (hit.priceNis == null) return "";
  const parts = [`₪${hit.priceNis.toFixed(2)}`];
  if (hit.priceMaxNis != null && hit.priceMaxNis > hit.priceNis + 0.009) {
    parts.push(`עד ₪${hit.priceMaxNis.toFixed(2)}`);
  }
  if (hit.priceAvgNis != null) parts.push(`ממוצע ₪${hit.priceAvgNis.toFixed(2)}`);
  if (hit.cheapestChain) parts.push(`הכי זול: ${hit.cheapestChain}`);
  if (hit.unitQty) parts.push(hit.unitQty);
  if (hit.priceStoreCount != null && hit.priceStoreCount > 0) {
    parts.push(`${hit.priceStoreCount} חנויות`);
  }
  return parts.join(" · ");
};

const applyPricePayload = (hit: ProductSerpHit, payload: CheapersalPricesPayload): ProductSerpHit => {
  const data = payload.data;
  if (!payload.success || !data?.summary?.cheapest) return hit;

  const summary = data.summary;
  const cheapestChain = summary.cheapestChain?.name?.trim();
  const priceSummary = formatProductPriceSummary({
    ...hit,
    priceNis: summary.cheapest,
    priceMaxNis: summary.mostExpensive,
    priceAvgNis: summary.average,
    cheapestChain,
    priceStoreCount: summary.storeCount,
    unitQty: data.product?.unitQty,
  });

  return {
    ...hit,
    priceNis: summary.cheapest,
    priceMaxNis: summary.mostExpensive,
    priceAvgNis: summary.average,
    cheapestChain,
    priceStoreCount: summary.storeCount,
    unitQty: data.product?.unitQty,
    priceSummary,
    imageUrl:
      stripGenericCatalogImage(hit.imageUrl) ||
      data.product?.image?.trim() ||
      data.product?.imageUrl?.trim() ||
      hit.imageUrl,
    url: supermarketProductUrl(hit.barcode, cheapestChain),
    snippet: priceSummary ? `${priceSummary} · ${hit.snippet}` : hit.snippet,
    source: hit.source.includes("Cheapersal") ? hit.source : `${hit.source} · Cheapersal`,
  };
};

export const fetchCheapersalPrices = async (barcode: string): Promise<CheapersalPricesPayload | null> => {
  if (!isCheapersalConfigured()) return null;
  const base = cheapersalBaseUrl();
  const url = `${base}/products/${encodeURIComponent(barcode)}/prices`;
  try {
    return await fetchJson<CheapersalPricesPayload>(url, undefined, {
      timeoutMs: 12_000,
      headers: cheapersalHeaders(),
    });
  } catch {
    return null;
  }
};

/** Enrich product hits with API prices and/or public Cheapersal product page (image + price). */
export const enrichProductHitsWithPrices = async (
  hits: ProductSerpHit[],
  limit = 8,
): Promise<ProductSerpHit[]> => {
  if (!hits.length) return hits;

  const targets = hits.slice(0, limit);
  const priced = await Promise.all(
    targets.map(async (hit) => {
      let next: ProductSerpHit = {
        ...hit,
        imageUrl: stripGenericCatalogImage(hit.imageUrl),
      };

      if (isCheapersalConfigured()) {
        const payload = await fetchCheapersalPrices(hit.barcode);
        if (payload) next = applyPricePayload(next, payload);
      }

      const pageUrl = isCheapersalProductUrl(hit.url) ? hit.url : cheapersalProductUrl(hit.barcode);
      const pub =
        (await fetchCheapersalPageMeta(pageUrl)) ?? (await fetchCheapersalPublicMeta(hit.barcode));
      if (pub) next = applyPublicMetaToHit(next, pub);

      if (!next.imageUrl || isGenericCatalogImage(next.imageUrl)) {
        const resolvedImage = await resolveProductImageUrl(hit.barcode, next.imageUrl, pageUrl);
        if (resolvedImage) next = { ...next, imageUrl: resolvedImage };
      }

      return next;
    }),
  );

  const pricedByBarcode = new Map(priced.map((h) => [h.barcode, h]));
  return sortProductHitsByRichness(hits.map((h) => pricedByBarcode.get(h.barcode) ?? h));
};
