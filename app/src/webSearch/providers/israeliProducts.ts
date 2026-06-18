import catalogJson from "../data/israeliProductCatalog.json";
import { fetchJson } from "../fetchJson";
import { buildPriceSearchQuery, buildProductSearchQuery, isPriceQuery } from "../intents";
import type { ProductSerpHit, SearchSourceResult } from "../types";
import {
  enrichProductHitsWithPrices,
  formatProductPriceSummary,
  isCheapersalConfigured,
} from "./cheapersalPrices";
import {
  supermarketProductUrl,
} from "./israeliProductLinks";
import { stripGenericCatalogImage } from "./productImageResolve";

type CatalogEntry = { name: string; brand?: string; category?: string };

const CATALOG = catalogJson as Record<string, CatalogEntry>;

const CATEGORY_HE: Record<string, string> = {
  fridge: "מקרר",
  freezer: "מקפיא",
  pantry: "מזווה",
  cleaning: "ניקיון",
};

const productPageUrl = (barcode: string, cheapestChain?: string): string =>
  supermarketProductUrl(barcode, cheapestChain);

const normalize = (text: string): string =>
  text
    .toLowerCase()
    .replace(/['׳"״]/g, "")
    .replace(/\s+/g, " ")
    .trim();

const scoreCatalogMatch = (query: string, entry: CatalogEntry): number => {
  const q = normalize(query);
  const name = normalize(entry.name);
  const brand = normalize(entry.brand ?? "");
  const tokens = q.split(" ").filter((t) => t.length >= 2);
  if (!tokens.length) return 0;

  let score = 0;
  for (const t of tokens) {
    if (name.includes(t)) score += 3;
    if (brand.includes(t)) score += 4;
    if (name.startsWith(t)) score += 1;
  }
  if (brand && q.includes(brand)) score += 5;
  if (name && q.includes(name)) score += 6;
  return score;
};

const searchLocalCatalog = (query: string, limit = 12): ProductSerpHit[] => {
  const scored: Array<{ score: number; hit: ProductSerpHit }> = [];
  for (const [barcode, entry] of Object.entries(CATALOG)) {
    const score = scoreCatalogMatch(query, entry);
    if (score < 3) continue;
    const title = entry.brand ? `${entry.name} — ${entry.brand}` : entry.name;
    const catHe = entry.category ? CATEGORY_HE[entry.category] || entry.category : "";
    scored.push({
      score,
      hit: {
        id: `catalog-${barcode}`,
        barcode,
        title,
        brand: entry.brand,
        category: entry.category,
        url: productPageUrl(barcode),
        snippet: [catHe, `ברקוד ${barcode}`, entry.brand ? `מותג ${entry.brand}` : ""]
          .filter(Boolean)
          .join(" · "),
        imageUrl: undefined,
        source: "Cheapersal · השוואת מחירים",
      },
    });
  }
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s) => s.hit);
};

type OffSearchProduct = {
  code?: string;
  product_name?: string;
  product_name_he?: string;
  brands?: string;
  image_front_small_url?: string;
  image_front_url?: string;
  categories_tags?: string[];
};

const mapOffProduct = (p: OffSearchProduct): ProductSerpHit | null => {
  const barcode = p.code?.trim();
  if (!barcode) return null;
  const nameHe = p.product_name_he?.trim();
  const nameEn = p.product_name?.trim();
  const brand = p.brands?.split(",")[0]?.trim();
  const title =
    nameHe && brand && !nameHe.includes(brand)
      ? `${nameHe} — ${brand}`
      : nameHe || nameEn || brand || `מוצר ${barcode}`;
  return {
    id: `off-${barcode}`,
    barcode,
    title,
    brand,
    url: productPageUrl(barcode),
    snippet: [brand ? `מותג ${brand}` : "", `ברקוד ${barcode}`, "מחירים בסופרמרקט · ישראל"]
      .filter(Boolean)
      .join(" · "),
    imageUrl: p.image_front_small_url || p.image_front_url || undefined,
    source: "Cheapersal · השוואת מחירים",
  };
};

async function searchOpenFoodFacts(query: string): Promise<ProductSerpHit[]> {
  const params = new URLSearchParams({
    search_terms: query,
    tagtype_0: "countries",
    tag_contains_0: "contains",
    tag_0: "israel",
    page_size: "15",
    fields: "code,product_name,product_name_he,brands,image_front_small_url,image_front_url",
    json: "1",
  });
  const data = await fetchJson<{ products?: OffSearchProduct[] }>(
    `https://world.openfoodfacts.org/cgi/search.pl?${params}`,
    undefined,
    { timeoutMs: 12_000 },
  );
  return (data.products ?? []).map(mapOffProduct).filter((h): h is ProductSerpHit => h != null);
}

async function lookupBarcode(barcode: string): Promise<ProductSerpHit | null> {
  const local = CATALOG[barcode];
  if (local) {
    const title = local.brand ? `${local.name} — ${local.brand}` : local.name;
    return {
      id: `catalog-${barcode}`,
      barcode,
      title,
      brand: local.brand,
      category: local.category,
      url: productPageUrl(barcode),
      snippet: `ברקוד ${barcode} · השוואת מחירים`,
      imageUrl: undefined,
      source: "Cheapersal · השוואת מחירים",
    };
  }

  try {
    const data = await fetchJson<{
      status?: number;
      product?: {
        product_name?: string;
        product_name_he?: string;
        brands?: string;
        image_front_small_url?: string;
        image_front_url?: string;
      };
    }>(
      `https://world.openfoodfacts.org/api/v2/product/${barcode}.json?fields=product_name,product_name_he,brands,image_front_url,image_front_small_url`,
      undefined,
      { timeoutMs: 10_000 },
    );
    if (data.status !== 1 || !data.product) return null;
    return mapOffProduct({
      code: barcode,
      product_name: data.product.product_name,
      product_name_he: data.product.product_name_he,
      brands: data.product.brands,
      image_front_small_url: data.product.image_front_small_url,
      image_front_url: data.product.image_front_url,
    });
  } catch {
    return null;
  }
}

const dedupeByBarcode = (hits: ProductSerpHit[]): ProductSerpHit[] => {
  const byCode = new Map<string, ProductSerpHit>();
  for (const hit of hits) {
    const prev = byCode.get(hit.barcode);
    if (!prev || hit.snippet.length > prev.snippet.length) {
      byCode.set(hit.barcode, hit);
    }
  }
  return [...byCode.values()];
};

const enrichImages = (hits: ProductSerpHit[]): ProductSerpHit[] =>
  hits.map((h) => ({
    ...h,
    imageUrl: stripGenericCatalogImage(h.imageUrl),
    url: h.url.includes("openfoodfacts.org")
      ? productPageUrl(h.barcode, h.cheapestChain)
      : h.cheapestChain
        ? productPageUrl(h.barcode, h.cheapestChain)
        : h.url || productPageUrl(h.barcode),
  }));

const formatHitsText = (hits: ProductSerpHit[], query: string, priceMode: boolean): string => {
  const lines = [`שאילתה: ${query}`];
  const top = hits[0];
  if (priceMode && top?.priceNis != null) {
    lines.push(`ANSWER: ${top.title} — ${formatProductPriceSummary(top)}`);
  } else if (priceMode && !isCheapersalConfigured()) {
    lines.push(
      "ANSWER: נמצאו מוצרים — מחירי סופרמרקט דורשים מפתח Cheapersal (CHEAPERSAL_API_KEY ב-app/.env).",
    );
  }
  hits.forEach((h, i) => {
    const price = h.priceNis != null ? ` · ${formatProductPriceSummary(h)}` : "";
    lines.push(`${i + 1}. ${h.title} [${h.barcode}]${price} — ${h.source} (${h.url})`);
  });
  return lines.join("\n");
};

const resolveProductQuery = (raw: string): string =>
  isPriceQuery(raw) ? buildPriceSearchQuery(raw) : buildProductSearchQuery(raw);

const isBarcodeQuery = (q: string): boolean => /^\d{8,14}$/.test(q.trim());

const shouldFetchPrices = (raw: string): boolean =>
  isPriceQuery(raw) || isBarcodeQuery(raw);

const finalizeHits = async (
  hits: ProductSerpHit[],
  raw: string,
): Promise<ProductSerpHit[]> => {
  const enriched = enrichImages(hits);
  if (!enriched.length) return enriched;
  const limit = isBarcodeQuery(raw) ? 1 : Math.min(enriched.length, 12);
  return enrichProductHitsWithPrices(enriched, limit);
};

export const fetchIsraeliProductsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "israeli-products" as const;
  const label = "מוצרי סופר · ישראל";
  const raw = query.trim();

  if (isBarcodeQuery(raw)) {
    const hit = await lookupBarcode(raw);
    if (!hit) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצא מוצר לברקוד ${raw}.`,
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const hits = await finalizeHits([hit], raw);
    const priceMode = shouldFetchPrices(raw);
    const priceWarning =
      priceMode && !hits[0]?.priceNis && isPriceQuery(raw)
        ? isCheapersalConfigured()
          ? "\nהערה: לא נמצאו מחירים במאגר Cheapersal למוצר זה."
          : "\nהערה: מחירי סופר דורשים CHEAPERSAL_API_KEY ב-app/.env (חינם: cheapersal.co.il/developers)."
        : "";
    return {
      provider,
      label,
      ok: true,
      text: formatHitsText(hits, raw, priceMode) + priceWarning,
      url: hits[0]?.url,
      productHits: hits,
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const pq = resolveProductQuery(raw);
  if (!pq || pq.length < 2) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "הקלד שם מוצר (למשל: חלב תנובה) או ברקוד ישראלי (729…).",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const [local, off] = await Promise.all([
      Promise.resolve(searchLocalCatalog(pq)),
      searchOpenFoodFacts(pq).catch(() => [] as ProductSerpHit[]),
    ]);

    const merged = dedupeByBarcode([...local, ...off]).slice(0, 12);
    const hits = await finalizeHits(merged, raw);
    const priceMode = shouldFetchPrices(raw);

    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצאו מוצרים עבור «${pq}». נסה שם מדויק יותר או ברקוד.`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const priceWarning =
      priceMode && isPriceQuery(raw) && !hits.some((h) => h.priceNis != null)
        ? isCheapersalConfigured()
          ? "\nהערה: לא נמצאו מחירים במאגר Cheapersal — נסה שם מדויק יותר."
          : "\nהערה: מחירי סופר דורשים CHEAPERSAL_API_KEY ב-app/.env (חינם: cheapersal.co.il/developers)."
        : "";

    return {
      provider,
      label,
      ok: true,
      text: formatHitsText(hits, pq, priceMode) + priceWarning,
      url: hits[0]?.url,
      productHits: hits,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: `שגיאה זמנית בחיפוש מוצרים עבור «${pq}».`,
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
