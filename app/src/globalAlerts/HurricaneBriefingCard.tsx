import type { CSSProperties } from "react";
import type { EnrichedStormBriefing } from "./enrichStormBriefing";
import { hurricaneColorCss } from "./hurricaneIntensity";
import { SeverityMeter } from "./SeverityMeter";
import type { GlobeAlertEvent } from "./types";
import { timeAgo } from "./useGlobalAlertEvents";

type Props = {
  ev: GlobeAlertEvent;
  enriched: EnrichedStormBriefing | null;
  loading: boolean;
  onReturn: () => void;
};

export function HurricaneBriefingCard({ ev, enriched, loading, onReturn }: Props) {
  const accent = hurricaneColorCss(ev.category, ev.severityText);

  return (
    <div className="ga-hurricane-brief" style={{ borderColor: accent, "--hb-accent": accent } as CSSProperties}>
      <div className="ga-hurricane-brief__scan" aria-hidden />
      <div className="ga-hurricane-brief__header">
        <div>
          <span className="ga-hurricane-brief__live">● LIVE TRACK</span>
          <h3 className="ga-hurricane-brief__title" style={{ color: accent }}>
            {enriched?.headline ?? ev.location}
          </h3>
        </div>
        <button type="button" className="ga-hurricane-brief__back" onClick={onReturn}>
          חזרה
        </button>
      </div>

      {loading ? (
        <p className="ga-hurricane-brief__loading">מאתר מיקום · מסלול · אוכלוסיה בסיכון…</p>
      ) : enriched ? (
        <>
          <p className="ga-hurricane-brief__hero">{enriched.currentRegion}</p>
          <p className="ga-hurricane-brief__move">{enriched.movementLine}</p>

          <ul className="ga-hurricane-brief__feed">
            {enriched.narrativeLines.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>

          {enriched.populationLabel ? (
            <div className="ga-hurricane-brief__pop" style={{ borderColor: accent }}>
              <span className="ga-hurricane-brief__pop-label">אוכלוסיה בסיכון</span>
              <span className="ga-hurricane-brief__pop-val">{enriched.populationLabel}</span>
            </div>
          ) : null}
        </>
      ) : (
        <p className="ga-hurricane-brief__loading">לא ניתן לטעון תדרוך מלא — GDACS/OpenStreetMap</p>
      )}

      <div className="ga-hurricane-brief__foot">
        <SeverityMeter ev={ev} color={accent} compact />
        <span className="ga-hurricane-brief__meta">
          {timeAgo(ev.time)} · GDACS · NOAA
        </span>
      </div>
    </div>
  );
}
