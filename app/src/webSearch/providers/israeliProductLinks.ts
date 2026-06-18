/** Public product pages — supermarket / price comparison, not barcode databases. */

export const cheapersalProductUrl = (barcode: string): string =>
  `https://cheapersal.co.il/product/${encodeURIComponent(barcode.trim())}`;

/** Same CDN Cheapersal uses on product pages (additlist catalog). */
export const cheapersalCatalogImage = (barcode: string): string =>
  `https://price-api.additlist.com/images/catalog/carrefour/${barcode.trim()}.jpg`;

export const ADDITLIST_IMAGE_CHAINS = [
  "carrefour",
  "rami-levy",
  "victory",
  "mega",
  "shufersal",
  "yohananof",
  "osher-ad",
  "tiv-taam",
] as const;

export const shufersalSearchUrl = (barcode: string): string =>
  `https://www.shufersal.co.il/online/he/search?text=${encodeURIComponent(barcode.trim())}`;

/** Best click-through URL: cheapest chain when known, otherwise Cheapersal price compare. */
export const supermarketProductUrl = (barcode: string, cheapestChain?: string): string => {
  const code = barcode.trim();
  const chain = (cheapestChain ?? "").trim();
  if (/רמי\s*לוי/i.test(chain)) {
    return `https://www.rami-levy.co.il/he/online/market/search?q=${encodeURIComponent(code)}`;
  }
  if (/שופרסל/i.test(chain)) return shufersalSearchUrl(code);
  if (/יוחננוף/i.test(chain)) {
    return `https://yochananof.co.il/?s=${encodeURIComponent(code)}`;
  }
  if (/ויקטורי|victory/i.test(chain)) {
    return `https://www.victoryonline.co.il/search?q=${encodeURIComponent(code)}`;
  }
  if (/קרפור|carrefour/i.test(chain)) {
    return `https://www.carrefour.co.il/search?text=${encodeURIComponent(code)}`;
  }
  if (/מגה|mega/i.test(chain)) {
    return `https://www.mega.co.il/online/search?q=${encodeURIComponent(code)}`;
  }
  return cheapersalProductUrl(code);
};

export const shufersalProductImage = (barcode: string): string =>
  `https://img.shufersal.co.il/imgs/Products_Vertical/${barcode.trim()}_V_large.jpg`;

export const ramiLevyProductImage = (barcode: string): string =>
  `https://static.rfrsh.co.il/supermarket/product/${barcode.trim()}/small.jpg`;

/** CDN fallbacks used by Israeli chains (kitchen-inventory order). */
export const productImageCandidates = (barcode: string, preferred?: string): string[] => {
  const code = barcode.trim();
  const additlist = ADDITLIST_IMAGE_CHAINS.map(
    (chain) => `https://price-api.additlist.com/images/catalog/${chain}/${code}.jpg`,
  );
  const candidates = [
    preferred?.trim(),
    ...additlist,
    cheapersalCatalogImage(code),
    shufersalProductImage(code),
    ramiLevyProductImage(code),
  ].filter((u): u is string => Boolean(u));
  return [...new Set(candidates)];
};
