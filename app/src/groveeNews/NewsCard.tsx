import { useState, type MouseEvent } from "react";
import { readAndSummarizeArticle } from "./bridge";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import { needsDisplayTranslation } from "./engine/summarize/languageDetect";
import { armOnDemandDeepRead } from "./summarizerApp";
import { useArticleImage } from "./engine/hooks/useArticleImage";
import { NewsSummaryTokenHud } from "./NewsSummaryTokenHud";
import type { GroveeNewsCard, NewsSummaryGemmaProgress } from "./types";

type Props = {
  card: GroveeNewsCard;
  expanded: boolean;
  onToggle: () => void;
  /** Gemma polish: receives Qwen draft, returns final Hebrew for card + chat. */
  onSummaryReady?: (
    card: GroveeNewsCard,
    qwenDraft: string,
    fallbackHe: string,
    progress?: NewsSummaryGemmaProgress,
  ) => Promise<string>;
};

const PHASE_COPY = {
  qwen: {
    title: "שלב 1 · Qwen",
    detail: "קורא את הכתבה ומפיק עובדות ותקציר ראשוני באנגלית",
  },
  gemma: {
    title: "שלב 2 · Gemma",
    detail: "מנסח מחדש בעברית ברורה לתצוגה בכרטיסיה ובצ'אט",
  },
} as const;

export function NewsCard({ card, expanded, onToggle, onSummaryReady }: Props) {
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<"qwen" | "gemma" | null>(null);
  const [qwenTokens, setQwenTokens] = useState(0);
  const [gemmaTokens, setGemmaTokens] = useState(0);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { src, state } = useArticleImage({
    articleId: card.id,
    articleUrl: card.url,
    title: card.titleOriginal || card.title,
    stockHint: card.laneLabel || card.laneId || "",
    existing: card.image,
    description: card.summary || "",
    priority: expanded ? 2 : 1,
    allowStockFallback: true,
  });

  const showImage = Boolean(src) && state !== "unavailable";
  const uiLang = getUserNewsProfile().uiLanguage || "he";
  const snippet = card.summary?.trim() ?? "";
  const showSnippet =
    snippet.length > 0 && !needsDisplayTranslation(card.title, snippet, uiLang);
  const foreignSnippet =
    snippet.length > 0 && needsDisplayTranslation(card.title, snippet, uiLang);

  const runSummary = async (e: MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    setLoadingPhase("qwen");
    setQwenTokens(0);
    setGemmaTokens(0);
    setError(null);
    armOnDemandDeepRead();
    try {
      const result = await readAndSummarizeArticle(card.url, {
        onQwenToken: setQwenTokens,
      });
      if (result.error && !result.summaryHe) {
        setError(result.error);
        return;
      }

      let display = result.summaryHe;
      if (result.qwenDraft && onSummaryReady) {
        setLoadingPhase("gemma");
        display = await onSummaryReady(card, result.qwenDraft, result.summaryHe, {
          onGemmaToken: setGemmaTokens,
        });
      }
      setSummary(display);
    } catch (err) {
      setError(err instanceof Error ? err.message : "שגיאה");
    } finally {
      setLoading(false);
      setLoadingPhase(null);
    }
  };

  const openSource = (e: MouseEvent) => {
    e.stopPropagation();
    window.open(card.url, "_blank", "noopener,noreferrer");
  };

  const phaseCopy = loadingPhase ? PHASE_COPY[loadingPhase] : null;

  return (
    <article className={`news-card-wrap${expanded ? " news-card-wrap--open" : ""}`}>
      <button
        type="button"
        className="news-card"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        {showImage ? (
          <img className="news-card-img" src={src} alt="" loading="lazy" />
        ) : state === "loading" ? (
          <div className="news-card-img news-card-img--loading" aria-hidden="true" />
        ) : (
          <div className="news-card-img news-card-img--empty" aria-hidden="true">
            📰
          </div>
        )}
        <div className="news-card-main">
          {card.laneLabel ? (
            <p className="news-card-lane">
              {card.laneIcon} {card.laneLabel}
            </p>
          ) : null}
          <p className="news-card-title">{card.title}</p>
          <p className="news-card-meta">{card.source}</p>
        </div>
        <span className="news-card-chevron" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
      </button>

      {expanded ? (
        <div className="news-card-expand">
          {showSnippet ? <p className="news-card-snippet">{snippet}</p> : null}
          {foreignSnippet && !summary ? (
            <p className="news-card-snippet news-card-snippet--hint">
              כתבה בשפה זרה — לחץ «סכם כתבה» לתקציר בעברית
            </p>
          ) : null}

          <div className="news-card-actions">
            <button
              type="button"
              className="news-card-btn news-card-btn--primary news-card-actions__summarize"
              onClick={(e) => void runSummary(e)}
              disabled={loading}
            >
              {loading
                ? loadingPhase === "gemma"
                  ? "מנסח…"
                  : "מסכם…"
                : "סכם כתבה"}
            </button>

            <div className="news-card-actions__hud">
              <NewsSummaryTokenHud
                qwenTokens={qwenTokens}
                gemmaTokens={gemmaTokens}
                activeModel={loadingPhase}
                visible={loading}
              />
            </div>

            <button
              type="button"
              className="news-card-btn news-card-btn--ghost news-card-actions__source"
              onClick={openSource}
            >
              מקור ↗
            </button>
          </div>

          {loading && phaseCopy ? (
            <div className="news-card-phase" role="status" aria-live="polite">
              <div
                className={`news-card-phase-step${loadingPhase === "qwen" ? " news-card-phase-step--active" : ""}${loadingPhase === "gemma" && qwenTokens > 0 ? " news-card-phase-step--done" : ""}`}
              >
                <span className="news-card-phase-badge">1</span>
                <div>
                  <p className="news-card-phase-title">{PHASE_COPY.qwen.title}</p>
                  <p className="news-card-phase-detail">{PHASE_COPY.qwen.detail}</p>
                </div>
              </div>
              <div
                className={`news-card-phase-step${loadingPhase === "gemma" ? " news-card-phase-step--active" : ""}`}
              >
                <span className="news-card-phase-badge">2</span>
                <div>
                  <p className="news-card-phase-title">{PHASE_COPY.gemma.title}</p>
                  <p className="news-card-phase-detail">{PHASE_COPY.gemma.detail}</p>
                </div>
              </div>
            </div>
          ) : null}

          {summary ? <p className="news-card-summary">{summary}</p> : null}
          {error ? <p className="news-card-error">{error}</p> : null}
        </div>
      ) : null}
    </article>
  );
}
