import { createPortal } from "react-dom";
import { formatNeoCountdown } from "./neoEta";
import type { EnrichedNeoBriefing } from "./enrichNeoBriefing";
import { liveNeoMetrics } from "./neoLiveMetrics";
import type { NeoOrbitTrack } from "./neoTrack";
import {
  formatRotationPeriod,
  getShowcaseEntryForEvent,
} from "./neoShowcaseCatalog";
import {
  inferNeoVisualProfile,
  SHAPE_LABELS,
  sizeComparisonLabel,
  SPECTRAL_TYPES,
} from "./spaceObjectVisuals";
import type { GlobeAlertEvent } from "./types";

type DetailProps = {
  ev: GlobeAlertEvent;
  neoTrack?: NeoOrbitTrack | null;
  neoEnriched?: EnrichedNeoBriefing | null;
  loading?: boolean;
  visible: boolean;
  inline?: boolean;
  onReturn: () => void;
};

function hazardBadge(ev: GlobeAlertEvent, distLd: number): { text: string; color: string } {
  if (ev.showcaseNeo) return { text: "מחזורי", color: "#66aaff" };
  if (distLd < 0.2) return { text: "קריטי", color: "#ff4444" };
  if (distLd < 1 || ev.isPha) return { text: "מעקב", color: "#ff8844" };
  if (distLd < 5) return { text: "קרוב", color: "#ffcc44" };
  return { text: "בטוח", color: "#44ff88" };
}

export function SpaceNeoDetailPanel({
  ev,
  neoTrack,
  neoEnriched,
  loading,
  visible,
  inline,
  onReturn,
}: DetailProps) {
  if (!visible || ev.type !== "neo") return null;

  const live = liveNeoMetrics(ev, neoTrack);
  const profile = inferNeoVisualProfile(ev, ev.id.length);
  const spec = SPECTRAL_TYPES[profile.spectral];
  const badge = hazardBadge(ev, live.distLd);
  const showcase = getShowcaseEntryForEvent(ev);
  const vRel = ev.vRel ?? ev.vInf ?? live.speedKmS;
  const diamKm = live.diameterKm ?? ev.diameterKm;
  const distSunAu = showcase?.distSunAu ?? ev.showcaseDistSunAu;

  const panel = (
    <div
      className={`space-neo-panel${inline ? " space-neo-panel--inline" : ""}`}
      role="dialog"
      aria-label="פרטי עצם חלל"
    >
      <div className="space-neo-panel__head">
        <h3>
          {ev.location}
          <span className="space-neo-panel__badge" style={{ background: badge.color }}>
            {badge.text}
          </span>
        </h3>
        <button type="button" className="space-neo-panel__close" onClick={onReturn} aria-label="סגור">
          ✕
        </button>
      </div>

      {!ev.showcaseNeo ? (
        <div className="space-neo-panel__countdown" aria-live="polite">
          <span>זמן למעבר / קרבה</span>
          <time>{formatNeoCountdown(ev.approachTime ?? ev.time)}</time>
        </div>
      ) : null}

      {loading && !showcase ? (
        <p className="space-neo-panel__loading">טוען נתוני מסלול…</p>
      ) : (
        <div className="space-neo-panel__grid">
          <div className="space-neo-panel__field space-neo-panel__field--badge">
            <label>סוג ספקטרלי</label>
            <span>{spec.name}</span>
          </div>
          <div className="space-neo-panel__field">
            <label>צורה</label>
            <span>{SHAPE_LABELS[profile.shape]}</span>
          </div>
          <div className="space-neo-panel__field">
            <label>קוטר</label>
            <span>
              {diamKm != null
                ? diamKm >= 1
                  ? `${diamKm.toFixed(2)} ק"מ`
                  : `${(diamKm * 1000).toFixed(0)} מ"ר`
                : "—"}
            </span>
          </div>
          <div className="space-neo-panel__field">
            <label>מהירות מסלולית</label>
            <span>{vRel.toFixed(1)} ק"מ/ש</span>
          </div>
          {showcase ? (
            <>
              <div className="space-neo-panel__field">
                <label>תקופת סיבוב עצמי</label>
                <span>{formatRotationPeriod(showcase.rotH)}</span>
              </div>
              <div className="space-neo-panel__field">
                <label>אקסצנטריות מסלול</label>
                <span>{showcase.ecc.toFixed(3)}</span>
              </div>
              <div className="space-neo-panel__field">
                <label>נטיית מסלול</label>
                <span>{showcase.inc.toFixed(1)}°</span>
              </div>
              <div className="space-neo-panel__field">
                <label>מרחק מהשמש</label>
                <span>{showcase.distSunAu.toFixed(2)} AU</span>
              </div>
              <div className="space-neo-panel__field">
                <label>שנת גילוי</label>
                <span>{showcase.discovery}</span>
              </div>
              <div className="space-neo-panel__field">
                <label>השוואת גודל</label>
                <span>{sizeComparisonLabel(diamKm)}</span>
              </div>
            </>
          ) : (
            <>
              <div className="space-neo-panel__field">
                <label>מרחק עכשיו</label>
                <span>{live.distLd.toFixed(2)} LD</span>
              </div>
              <div className="space-neo-panel__field">
                <label>קרבה מינימלית</label>
                <span>{(ev.distLd ?? live.distLd).toFixed(2)} LD</span>
              </div>
              <div className="space-neo-panel__field">
                <label>ETA מעבר</label>
                <span>{formatNeoCountdown(ev.approachTime ?? ev.time)}</span>
              </div>
              <div className="space-neo-panel__field">
                <label>השוואת גודל</label>
                <span>{sizeComparisonLabel(diamKm)}</span>
              </div>
              <div className="space-neo-panel__field">
                <label>סיכון לציבור</label>
                <span style={{ color: neoEnriched?.publicRisk === "high" ? "#ff8844" : undefined }}>
                  {neoEnriched?.riskLabel ?? "—"}
                </span>
              </div>
            </>
          )}
          <div className="space-neo-panel__field space-neo-panel__field--full">
            <label>הרכב ותיאור</label>
            <span className="space-neo-panel__desc">
              {spec.desc}
              {showcase ? `. ${showcase.desc}` : ""}
              {!showcase && ev.isPha ? " · PHA — מעקב NASA פעיל." : ""}
              {!showcase && neoEnriched?.riskDetail ? ` ${neoEnriched.riskDetail}` : ""}
            </span>
          </div>
        </div>
      )}
    </div>
  );

  if (inline) return panel;

  return createPortal(
    <div className="space-neo-panel-anchor" dir="ltr">
      {panel}
    </div>,
    document.body,
  );
}

/** Inline alert detail — inside sidebar card for incoming NEOs. */
export function NeoAlertCardDetail({
  ev,
  neoEnriched,
  neoLive,
  loading,
}: {
  ev: GlobeAlertEvent;
  neoEnriched?: EnrichedNeoBriefing | null;
  neoLive?: ReturnType<typeof liveNeoMetrics> | null;
  loading?: boolean;
}) {
  if (ev.type !== "neo" || ev.showcaseNeo) return null;

  const profile = inferNeoVisualProfile(ev, ev.id.length);
  const spec = SPECTRAL_TYPES[profile.spectral];
  const live = neoLive ?? liveNeoMetrics(ev);
  const diamKm = live.diameterKm ?? ev.diameterKm;

  return (
    <div className="global-alerts-card__showcase-detail">
      {loading ? (
        <p className="space-neo-panel__loading">טוען נתוני מסלול…</p>
      ) : (
        <div className="global-alerts-card__showcase-grid">
          <div>
            <span className="global-alerts-card__showcase-label">סוג ספקטרלי</span>
            <span>{spec.name}</span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">צורה</span>
            <span>{SHAPE_LABELS[profile.shape]}</span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">קוטר</span>
            <span>
              {diamKm != null
                ? diamKm >= 1
                  ? `${diamKm.toFixed(2)} ק"מ`
                  : `${(diamKm * 1000).toFixed(0)} מ"ר`
                : "—"}
            </span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">מרחק עכשיו</span>
            <span>{live.distLd.toFixed(2)} LD</span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">קרבה מינימלית</span>
            <span>{(ev.distLd ?? live.distLd).toFixed(2)} LD</span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">מהירות</span>
            <span>{live.speedKmS.toFixed(1)} ק"מ/ש</span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">ETA מעבר</span>
            <span>{formatNeoCountdown(ev.approachTime ?? ev.time)}</span>
          </div>
          <div>
            <span className="global-alerts-card__showcase-label">סיכון לציבור</span>
            <span>{neoEnriched?.riskLabel ?? "בעדכון"}</span>
          </div>
          <div className="global-alerts-card__showcase-desc">
            <span className="global-alerts-card__showcase-label">אזור מעבר</span>
            <span>{neoEnriched?.impactRegion ?? "בעדכון"}</span>
          </div>
          {neoEnriched?.riskDetail ? (
            <div className="global-alerts-card__showcase-desc">
              <span className="global-alerts-card__showcase-label">הערכת סיכון</span>
              <span>{neoEnriched.riskDetail}</span>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

/** Inline catalog detail — lives inside the sidebar card, not a floating panel. */
export function ShowcaseCardDetail({ ev }: { ev: GlobeAlertEvent }) {
  if (!ev.showcaseNeo) return null;
  const showcase = getShowcaseEntryForEvent(ev);
  if (!showcase) return null;

  const profile = inferNeoVisualProfile(ev, ev.id.length);
  const spec = SPECTRAL_TYPES[profile.spectral];
  const diamKm = ev.diameterKm ?? showcase.diamM / 1000;

  return (
    <div className="global-alerts-card__showcase-detail">
      <div className="global-alerts-card__showcase-grid">
        <div>
          <span className="global-alerts-card__showcase-label">סוג ספקטרלי</span>
          <span>{spec.name}</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">צורה</span>
          <span>{SHAPE_LABELS[profile.shape]}</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">קוטר</span>
          <span>
            {diamKm >= 1 ? `${diamKm.toFixed(2)} ק"מ` : `${(diamKm * 1000).toFixed(0)} מ"ר`}
          </span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">מרחק מינימלי</span>
          <span>{showcase.distLd.toFixed(1)} LD</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">מהירות</span>
          <span>{showcase.vRel.toFixed(1)} ק"מ/ש</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">סיבוב עצמי</span>
          <span>{formatRotationPeriod(showcase.rotH)}</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">אקסצנטריות</span>
          <span>{showcase.ecc.toFixed(3)}</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">נטיית מסלול</span>
          <span>{showcase.inc.toFixed(1)}°</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">מרחק מהשמש</span>
          <span>{showcase.distSunAu.toFixed(2)} AU</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">שנת גילוי</span>
          <span>{showcase.discovery}</span>
        </div>
        <div>
          <span className="global-alerts-card__showcase-label">השוואת גודל</span>
          <span>{sizeComparisonLabel(diamKm)}</span>
        </div>
        <div className="global-alerts-card__showcase-desc">
          <span className="global-alerts-card__showcase-label">תיאור</span>
          <span>{showcase.desc}</span>
        </div>
      </div>
    </div>
  );
}
