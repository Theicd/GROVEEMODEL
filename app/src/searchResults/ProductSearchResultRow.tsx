import { useMemo, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { productImageCandidates } from "../webSearch/providers/israeliProductLinks";
import { isGenericCatalogImage } from "../webSearch/providers/productImageResolve";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
};

const labels = {
  he: { pill: "מוצר", open: "פתח", from: "החל מ-", noImage: "אין תמונה", noPrice: "מחיר לא זמין ב-Cheapersal" },
  en: { pill: "Product", open: "Open", from: "from ", noImage: "No image", noPrice: "Price not on Cheapersal" },
} as const;

const productBarcode = (hit: UnifiedSearchHit): string => hit.meta?.engine?.trim() ?? "";

const formatPriceBlock = (hit: UnifiedSearchHit, uiLang: ChatUiLanguage): string => {
  if (hit.meta?.priceNis == null) return "";
  const L = labels[uiLang];
  const parts = [`₪${hit.meta.priceNis.toFixed(2)}`];
  const raw = hit.snippetOriginal ?? hit.snippet;
  const maxMatch = raw.match(/עד ₪([\d.]+)/);
  if (maxMatch) parts.push(`${uiLang === "he" ? "עד" : "up to"} ₪${maxMatch[1]}`);
  if (raw.includes("הכי זול")) {
    const chain = raw.match(/הכי זול:\s*([^·]+)/)?.[1]?.trim();
    if (chain) parts.push(`${uiLang === "he" ? "הכי זול" : "cheapest"}: ${chain}`);
  }
  return `${L.from}${parts.join(" · ")}`;
};

/** Supermarket product — inline image + live Cheapersal price. */
export function ProductSearchResultRow({ hit, uiLang }: Props) {
  const L = labels[uiLang];
  const barcode = productBarcode(hit);
  const preferred = hit.imageUrl && !isGenericCatalogImage(hit.imageUrl) ? hit.imageUrl : undefined;
  const candidates = useMemo(
    () => productImageCandidates(barcode, preferred),
    [barcode, preferred],
  );
  const [imgIdx, setImgIdx] = useState(0);
  const [imgExhausted, setImgExhausted] = useState(false);
  const imgSrc = !imgExhausted ? candidates[imgIdx] : undefined;
  const priceLine = formatPriceBlock(hit, uiLang);
  const detailSnippet =
    hit.snippet && hit.meta?.priceNis != null
      ? hit.snippet.replace(/^₪[\d.]+(?:\s*·\s*)?/, "").replace(/עד ₪[\d.]+\s*·\s*/, "").trim()
      : hit.snippet;

  const onImgError = () => {
    if (imgIdx + 1 < candidates.length) {
      setImgIdx((i) => i + 1);
    } else {
      setImgExhausted(true);
    }
  };

  return (
    <article className="serp-row serp-row--media-inline serp-row--product-inline" dir={uiLang === "he" ? "rtl" : "ltr"}>
      <div className="serp-row-site">
        <div className="serp-row-site-main">
          <span className="serp-row-site-name">{hit.sourceLabel}</span>
        </div>
        <span className="serp-product-pill">{L.pill}</span>
      </div>

      <div className="serp-media-inline-body">
        <a
          className="serp-media-inline-thumb-btn"
          href={hit.url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${L.open}: ${hit.title}`}
        >
          <span className="serp-media-inline-thumb-wrap serp-media-inline-thumb-wrap--product">
            {imgSrc ? (
              <img
                key={imgSrc}
                className="serp-media-inline-thumb"
                src={imgSrc}
                alt=""
                loading="lazy"
                referrerPolicy="no-referrer"
                onError={onImgError}
              />
            ) : (
              <span className="serp-media-inline-thumb serp-media-inline-thumb--placeholder" aria-hidden="true">
                <span className="serp-product-thumb-fallback">{L.noImage}</span>
              </span>
            )}
          </span>
        </a>

        <div className="serp-media-inline-text">
          <a className="serp-row-title" href={hit.url} target="_blank" rel="noopener noreferrer" title={hit.title}>
            {hit.title}
          </a>
          {priceLine ? (
            <p className="serp-row-product-price serp-product-inline-price" dir="ltr">
              {priceLine}
            </p>
          ) : (
            <p className="serp-row-snippet serp-product-no-price">{L.noPrice}</p>
          )}
          {detailSnippet ? <p className="serp-row-snippet serp-media-inline-snippet">{detailSnippet}</p> : null}
          <div className="serp-media-inline-actions">
            <a className="serp-btn" href={hit.url} target="_blank" rel="noopener noreferrer">
              {L.open}
            </a>
            {barcode ? (
              <span className="serp-btn serp-btn--ghost" dir="ltr">
                {barcode}
              </span>
            ) : null}
          </div>
        </div>
      </div>
    </article>
  );
}

export const isProductHit = (hit: UnifiedSearchHit): boolean => hit.kind === "product";
