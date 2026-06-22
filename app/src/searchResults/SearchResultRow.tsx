import { useState, type MouseEvent } from "react";
import { readAndSummarizeArticle } from "../groveeNews/bridge";
import type { GroveeNewsCard, NewsSummaryGemmaProgress } from "../groveeNews/types";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { displayBreadcrumb, googleTranslatePageUrl } from "./displayUrl";
import { formatGithubTitleLine } from "./formatGithubHit";
import { hitNeedsTranslatePageLink } from "./translateHits";
import { hostFromUrl } from "./sourceBranding";
import { webProviderLabel } from "./webProviderLabels";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
  onSummaryReady?: (
    card: GroveeNewsCard,
    gemmaInput: string,
    progress?: NewsSummaryGemmaProgress,
  ) => Promise<string>;
};

const hitToCard = (hit: UnifiedSearchHit): GroveeNewsCard => ({
  id: hit.id,
  title: hit.titleOriginal ?? hit.title,
  titleOriginal: hit.titleOriginal ?? hit.title,
  source: hit.sourceLabel,
  sourceKey: hit.sourceKey ?? hit.sourceLabel,
  url: hit.url,
  image: hit.imageUrl ?? "",
  score: hit.score ?? 0,
  publishedTs: hit.publishedTs ?? 0,
  summary: hit.snippetOriginal ?? hit.snippet,
});

const siteNameForHit = (hit: UnifiedSearchHit): string => {
  if (hit.kind === "rss") return hit.sourceLabel;
  if (hit.kind === "movie") return hit.sourceLabel;
  if (hit.kind === "web") return hit.sourceLabel || hostFromUrl(hit.url);
  return hit.sourceLabel;
};

const formatProductPriceLine = (hit: UnifiedSearchHit): string => {
  const raw = hit.snippetOriginal ?? hit.snippet;
  if (raw.includes("₪")) {
    const head = raw.split(" · ")[0]?.trim() ?? "";
    if (head.startsWith("₪")) return head;
  }
  if (hit.meta?.priceNis != null) return `₪${hit.meta.priceNis.toFixed(2)}`;
  return "";
};

const labels = {
  he: {
    summarize: "סכם כתבה",
    summarizing: "מסכם…",
    translatePage: "לדף המתורגם",
    summarizeError: "לא ניתן לסכם את הכתבה",
  },
  en: {
    summarize: "Summarize",
    summarizing: "Summarizing…",
    translatePage: "Translate page",
    summarizeError: "Could not summarize article",
  },
} as const;

export function SearchResultRow({ hit, uiLang, onSummaryReady }: Props) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const L = labels[uiLang];
  const siteName = siteNameForHit(hit);
  const favicon = hit.faviconUrl;
  const poster = hit.kind === "movie" ? hit.imageUrl : undefined;
  const showTranslate = hitNeedsTranslatePageLink(hit, uiLang);
  const translateUrl = googleTranslatePageUrl(hit.url, uiLang);
  const githubTitle = hit.kind === "github" ? formatGithubTitleLine(hit.title) : null;

  const runSummary = async (e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!hit.summarizable) return;
    setLoading(true);
    setError(null);
    try {
      const result = await readAndSummarizeArticle(hit.url);
      if (result.error && !result.gemmaInput) {
        setError(result.error);
        return;
      }
      let display = uiLang === "he" ? result.summaryHe : result.summaryHe;
      if (result.gemmaInput && onSummaryReady) {
        display = await onSummaryReady(hitToCard(hit), result.gemmaInput);
      } else if (!display) {
        setError(L.summarizeError);
        return;
      }
      setSummary(display);
    } catch (err) {
      setError(err instanceof Error ? err.message : uiLang === "he" ? "שגיאה" : "Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <article className={`serp-row${hit.kind === "movie" ? " serp-row--movie" : ""}`} dir={uiLang === "he" ? "rtl" : "ltr"}>
      <div className="serp-row-site">
        <div className="serp-row-site-main">
          {poster ? (
            <img className="serp-row-poster" src={poster} alt="" width={36} height={54} loading="lazy" />
          ) : hit.kind === "product" && hit.imageUrl ? (
            <img
              className="serp-row-poster serp-row-poster--product"
              src={hit.imageUrl}
              alt=""
              width={36}
              height={36}
              loading="lazy"
              referrerPolicy="no-referrer"
              onError={(e) => {
                const img = e.currentTarget;
                const fallback = img.dataset.fallback;
                if (fallback && img.src !== fallback) {
                  img.src = fallback;
                  return;
                }
                img.style.visibility = "hidden";
              }}
              data-fallback={
                hit.meta?.engine
                  ? `https://static.rfrsh.co.il/supermarket/product/${hit.meta.engine}/small.jpg`
                  : undefined
              }
            />
          ) : favicon ? (
            <img className="serp-row-favicon" src={favicon} alt="" width={18} height={18} loading="lazy" />
          ) : (
            <span className="serp-row-favicon serp-row-favicon--placeholder" aria-hidden="true" />
          )}
          <span className="serp-row-site-name">{siteName}</span>
          {hit.kind === "github" && hit.meta?.engine ? (
            <span className="serp-row-meta-inline">{hit.meta.engine}</span>
          ) : null}
          {hit.kind === "movie" && hit.meta?.engine ? (
            <span className="serp-row-meta-inline">{hit.meta.engine}</span>
          ) : null}
          {hit.kind === "movie" && hit.meta?.stars != null ? (
            <span className="serp-row-meta-inline">★{hit.meta.stars.toFixed(1)}</span>
          ) : null}
          {hit.kind === "product" && hit.meta?.priceNis != null ? (
            <span className="serp-row-meta-inline serp-row-price" dir="ltr">
              ₪{hit.meta.priceNis.toFixed(2)}
            </span>
          ) : null}
          {hit.kind === "product" && hit.meta?.engine && hit.meta?.priceNis == null ? (
            <span className="serp-row-meta-inline" dir="ltr">
              {hit.meta.engine}
            </span>
          ) : null}
          {hit.meta?.stars != null ? (
            <span className="serp-row-meta-inline">★{hit.meta.stars.toLocaleString("en-US")}</span>
          ) : null}
        </div>
        {hit.kind === "rss" ? <span className="serp-rss-pill">RSS</span> : null}
        {hit.kind === "web" && hit.provider === "openserp" ? (
          <span className="serp-companion-pill" title="Grove Search Companion">
            OpenSERP
          </span>
        ) : null}
        {hit.kind === "web" && hit.provider !== "openserp" && webProviderLabel(hit.provider, uiLang) ? (
          <span className="serp-web-pill">{webProviderLabel(hit.provider, uiLang)}</span>
        ) : null}
        {hit.kind === "movie" ? (
          <span className="serp-movie-pill">{uiLang === "he" ? "סרט" : "Movie"}</span>
        ) : null}
        {hit.kind === "product" ? (
          <span className="serp-product-pill">{uiLang === "he" ? "מוצר" : "Product"}</span>
        ) : null}
      </div>

      <div className="serp-row-url-line">
        <span className="serp-row-url" dir="ltr" title={hit.url}>
          {displayBreadcrumb(hit.url)}
        </span>
        {showTranslate ? (
          <a
            className="serp-row-translate"
            href={translateUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            {L.translatePage}
          </a>
        ) : null}
      </div>

      <a
        className="serp-row-title"
        href={hit.url}
        target="_blank"
        rel="noopener noreferrer"
        title={hit.title}
      >
        {githubTitle ? (
          <>
            <span className="serp-row-github-repo" dir="ltr">
              {githubTitle.repo}
            </span>
            {githubTitle.description ? (
              <>
                <span className="serp-row-github-sep">: </span>
                <span>{githubTitle.description}</span>
              </>
            ) : null}
          </>
        ) : (
          hit.title
        )}
      </a>

      {hit.kind === "product" && hit.meta?.priceNis != null ? (
        <p className="serp-row-product-price" dir="ltr">
          {formatProductPriceLine(hit)}
        </p>
      ) : null}

      {hit.snippet ? <p className="serp-row-snippet">{hit.snippet}</p> : null}

      {hit.summarizable ? (
        <div className="serp-row-actions">
          <button
            type="button"
            className="serp-btn"
            onClick={(e) => void runSummary(e)}
            disabled={loading}
          >
            {loading ? L.summarizing : L.summarize}
          </button>
        </div>
      ) : null}

      {summary ? <p className="serp-row-summary">{summary}</p> : null}
      {error ? <p className="serp-row-error">{error}</p> : null}
    </article>
  );
}
