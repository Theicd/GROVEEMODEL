import { NewsCardGrid } from "./NewsCardGrid";
import type { GroveeNewsCard, NewsPanelPayload, NewsSummaryGemmaProgress } from "./types";
import "./newsPanel.css";

type Props = {
  payload: NewsPanelPayload;
  onClose: () => void;
  onSummaryReady?: (
    card: GroveeNewsCard,
    qwenDraft: string,
    fallbackHe: string,
    progress?: NewsSummaryGemmaProgress,
  ) => Promise<string>;
};

export function NewsSidePanel({ payload, onClose, onSummaryReady }: Props) {
  const title =
    payload.mode === "topics"
      ? `Topics · ${payload.cards.length} כרטיסים`
      : `חיפוש · ${payload.query.slice(0, 48)}`;

  return (
    <div className="news-panel-inner">
      <header className="news-panel-head">
        <div className="news-panel-title">
          <span className="news-panel-dot" aria-hidden="true" />
          <span>{title}</span>
        </div>
        <button type="button" className="news-panel-close" onClick={onClose} aria-label="סגור חלונית חדשות">
          סגור
        </button>
      </header>
      <div className="news-panel-body news-panel-scroll">
        <NewsCardGrid cards={payload.cards} onSummaryReady={onSummaryReady} />
      </div>
    </div>
  );
}
