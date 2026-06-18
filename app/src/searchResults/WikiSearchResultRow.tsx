import { useState, type MouseEvent } from "react";
import { readAndSummarizeArticle } from "../groveeNews/bridge";
import type { GroveeNewsCard, NewsSummaryGemmaProgress } from "../groveeNews/types";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
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

const labels = {
  he: {
    pill: "ויקיפדיה",
    open: "פתח",
    summarize: "סכם כתבה",
    summarizing: "מסכם…",
    summarizeError: "לא ניתן לסכם את הכתבה",
  },
  en: {
    pill: "Wikipedia",
    open: "Open",
    summarize: "Summarize",
    summarizing: "Summarizing…",
    summarizeError: "Could not summarize article",
  },
} as const;

/** Wikipedia hit with infobox thumbnail — same inline media layout as Pixabay rows. */
export function WikiSearchResultRow({ hit, uiLang, onSummaryReady }: Props) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const L = labels[uiLang];

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
    <article className="serp-row serp-row--media-inline serp-row--wiki-inline" dir={uiLang === "he" ? "rtl" : "ltr"}>
      <div className="serp-row-site">
        <div className="serp-row-site-main">
          <span className="serp-row-site-name">{hit.sourceLabel}</span>
        </div>
        <span className="serp-wiki-pill">{L.pill}</span>
      </div>

      <div className="serp-media-inline-body">
        <a
          className="serp-media-inline-thumb-btn"
          href={hit.url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${L.open}: ${hit.title}`}
        >
          <span className="serp-media-inline-thumb-wrap serp-media-inline-thumb-wrap--portrait">
            {hit.imageUrl ? (
              <img
                className="serp-media-inline-thumb"
                src={hit.imageUrl}
                alt=""
                loading="lazy"
                referrerPolicy="no-referrer"
              />
            ) : (
              <span className="serp-media-inline-thumb serp-media-inline-thumb--placeholder" />
            )}
          </span>
        </a>

        <div className="serp-media-inline-text">
          <a className="serp-row-title" href={hit.url} target="_blank" rel="noopener noreferrer" title={hit.title}>
            {hit.title}
          </a>
          {hit.snippet ? <p className="serp-row-snippet serp-media-inline-snippet">{hit.snippet}</p> : null}
          <div className="serp-media-inline-actions">
            {hit.summarizable ? (
              <button type="button" className="serp-btn" onClick={(e) => void runSummary(e)} disabled={loading}>
                {loading ? L.summarizing : L.summarize}
              </button>
            ) : null}
            <a className="serp-btn serp-btn--ghost" href={hit.url} target="_blank" rel="noopener noreferrer">
              {L.open}
            </a>
          </div>
          {summary ? <p className="serp-row-summary">{summary}</p> : null}
          {error ? <p className="serp-row-error">{error}</p> : null}
        </div>
      </div>
    </article>
  );
}

export const isWikiHitWithImage = (hit: UnifiedSearchHit): boolean =>
  (hit.provider === "wikipedia-en" || hit.provider === "wikipedia-he") && Boolean(hit.imageUrl);
