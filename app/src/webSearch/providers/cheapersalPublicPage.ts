import { fetchText } from "../fetchJson";
import type { ProductSerpHit } from "../types";
import { cheapersalProductUrl } from "./israeliProductLinks";
import { stripGenericCatalogImage } from "./productImageResolve";
import { formatProductPriceSummary } from "./cheapersalPrices";

export type CheapersalPublicMeta = {
  imageUrl?: string;
  priceNis?: number;
  priceMaxNis?: number;
  priceAvgNis?: number;
};

const CHEAPERSAL_PRODUCT_RE = /cheapersal\.co\.il\/product\/\d+/i;

const parsePrice = (raw?: string | number): number | undefined => {
  if (raw == null || raw === "") return undefined;
  const n = typeof raw === "number" ? raw : Number.parseFloat(String(raw).replace(/,/g, ""));
  return Number.isFinite(n) && n > 0 ? n : undefined;
};

const normalizeImageUrl = (raw?: string): string | undefined => {
  const url = raw?.trim();
  if (!url) return undefined;
  if (url.startsWith("//")) return `https:${url}`;
  return url;
};

type JsonLdNode = Record<string, unknown>;

const isProductNode = (node: JsonLdNode): boolean => {
  const t = node["@type"];
  if (t === "Product") return true;
  if (Array.isArray(t)) return t.includes("Product");
  return false;
};

const parseJsonLdProduct = (html: string): CheapersalPublicMeta => {
  const out: CheapersalPublicMeta = {};
  const blocks = [...html.matchAll(/<script[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)];

  for (const [, raw] of blocks) {
    try {
      const data = JSON.parse(raw) as JsonLdNode | JsonLdNode[];
      const nodes = Array.isArray(data) ? data : [data];
      for (const node of nodes) {
        if (!isProductNode(node)) continue;

        const image = node.image;
        if (!out.imageUrl) {
          if (typeof image === "string") out.imageUrl = normalizeImageUrl(image);
          else if (Array.isArray(image) && typeof image[0] === "string") {
            out.imageUrl = normalizeImageUrl(image[0]);
          }
        }

        const offers = node.offers as JsonLdNode | JsonLdNode[] | undefined;
        const offer = Array.isArray(offers) ? offers[0] : offers;
        if (offer && typeof offer === "object") {
          const low = parsePrice(offer.lowPrice as number | string | undefined);
          const high = parsePrice(offer.highPrice as number | string | undefined);
          const single = parsePrice(offer.price as number | string | undefined);
          if (low != null) out.priceNis = out.priceNis == null ? low : Math.min(out.priceNis, low);
          if (high != null) out.priceMaxNis = high;
          if (single != null && out.priceNis == null) out.priceNis = single;
        }
      }
    } catch {
      /* ignore malformed JSON-LD */
    }
  }

  return out;
};

const parseCheapersalPriceFallbacks = (html: string): Pick<CheapersalPublicMeta, "priceNis" | "priceMaxNis" | "priceAvgNis"> => {
  const titlePrice = html.match(/החל מ[-–]?\s*₪\s*([\d]+(?:\.[\d]{1,2})?)/i)?.[1];
  const faqPrice = html.match(/המחיר הזול ביותר[^₪]{0,120}₪\s*([\d]+(?:\.[\d]{1,2})?)/i)?.[1];
  const labeledLow = html.match(/המחיר הזול[^₪]{0,80}₪\s*([\d]+(?:\.[\d]{1,2})?)/i)?.[1];
  const labeledAvg = html.match(/מחיר ממוצע[^₪]{0,80}₪\s*([\d]+(?:\.[\d]{1,2})?)/i)?.[1];

  const allPrices = [...html.matchAll(/₪\s*([\d]+(?:\.[\d]{1,2})?)/g)]
    .map((m) => parsePrice(m[1]))
    .filter((n): n is number => n != null && n >= 0.5 && n < 500);

  const priceNis =
    parsePrice(titlePrice) ??
    parsePrice(faqPrice) ??
    parsePrice(labeledLow) ??
    (allPrices.length ? Math.min(...allPrices) : undefined);

  const priceMaxNis = allPrices.length ? Math.max(...allPrices) : undefined;
  const priceAvgNis = parsePrice(labeledAvg);

  return { priceNis, priceMaxNis, priceAvgNis };
};

/** Parse Cheapersal product HTML (SSR) for poster + cheapest price. */
export const parseCheapersalProductHtml = (html: string): CheapersalPublicMeta => {
  if (/מוצר לא נמצא|product not found/i.test(html)) {
    return {};
  }

  const fromLd = parseJsonLdProduct(html);
  const ogImage = normalizeImageUrl(html.match(/property=["']og:image["']\s+content=["']([^"']+)["']/i)?.[1]);
  const additlistImg = html.match(
    /https:\/\/price-api\.additlist\.com\/images\/[^"'\s<>]+\.(?:jpg|jpeg|webp|png)/i,
  )?.[0];

  const fallbacks = parseCheapersalPriceFallbacks(html);

  return {
    imageUrl: fromLd.imageUrl || ogImage || additlistImg || undefined,
    priceNis: fromLd.priceNis ?? fallbacks.priceNis,
    priceMaxNis: fromLd.priceMaxNis ?? fallbacks.priceMaxNis,
    priceAvgNis: fromLd.priceAvgNis ?? fallbacks.priceAvgNis,
  };
};

export const isCheapersalProductUrl = (url: string): boolean => CHEAPERSAL_PRODUCT_RE.test(url);

/** Fetch + parse the exact product page URL shown in search results. */
export const fetchCheapersalPageMeta = async (pageUrl: string): Promise<CheapersalPublicMeta | null> => {
  const url = pageUrl.trim();
  if (!isCheapersalProductUrl(url)) return null;
  try {
    const html = await fetchText(url, undefined, { timeoutMs: 14_000 });
    const meta = parseCheapersalProductHtml(html);
    if (!meta.imageUrl && meta.priceNis == null) return null;
    return meta;
  } catch {
    return null;
  }
};

export const fetchCheapersalPublicMeta = async (barcode: string): Promise<CheapersalPublicMeta | null> => {
  const code = barcode.trim();
  if (!/^\d{8,14}$/.test(code)) return null;
  return fetchCheapersalPageMeta(cheapersalProductUrl(code));
};

export const applyPublicMetaToHit = (
  hit: ProductSerpHit,
  meta: CheapersalPublicMeta,
): ProductSerpHit => {
  const priceNis = hit.priceNis ?? meta.priceNis;
  const priceMaxNis = hit.priceMaxNis ?? meta.priceMaxNis;
  const priceAvgNis = hit.priceAvgNis ?? meta.priceAvgNis;
  const imageUrl = meta.imageUrl || stripGenericCatalogImage(hit.imageUrl) || undefined;
  const priceSummary =
    hit.priceSummary ||
    (priceNis != null
      ? formatProductPriceSummary({
          ...hit,
          priceNis,
          priceMaxNis,
          priceAvgNis,
        })
      : undefined);
  const snippet =
    priceSummary && !hit.snippet.includes("₪")
      ? `${priceSummary} · ${hit.snippet}`
      : hit.snippet;

  return {
    ...hit,
    priceNis,
    priceMaxNis,
    priceAvgNis,
    imageUrl,
    priceSummary,
    snippet,
  };
};
