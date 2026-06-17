import { useState } from "react";
import { NewsCard } from "./NewsCard";
import type { GroveeNewsCard, NewsSummaryGemmaProgress } from "./types";
import "./newsPanel.css";

type Props = {
  cards: GroveeNewsCard[];
  onSummaryReady?: (
    card: GroveeNewsCard,
    qwenDraft: string,
    fallbackHe: string,
    progress?: NewsSummaryGemmaProgress,
  ) => Promise<string>;
};

export function NewsCardGrid({ cards, onSummaryReady }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (!cards.length) {
    return <div className="news-panel-empty">אין כרטיסיות להצגה — המאגר עדיין מתמלא ברקע.</div>;
  }

  return (
    <>
      {cards.map((card) => (
        <NewsCard
          key={card.id}
          card={card}
          expanded={expandedId === card.id}
          onToggle={() => setExpandedId((prev) => (prev === card.id ? null : card.id))}
          onSummaryReady={onSummaryReady}
        />
      ))}
    </>
  );
}
