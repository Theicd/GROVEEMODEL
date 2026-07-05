import type { CSSProperties } from "react";
import { useEffect, useState } from "react";
import type { EnrichedStormBriefing } from "./enrichStormBriefing";
import { hurricaneColorCss } from "./hurricaneIntensity";
import { formatNeoCountdown } from "./neoEta";
import type { EnrichedNeoBriefing } from "./enrichNeoBriefing";
import { liveNeoMetrics } from "./neoLiveMetrics";
import { approachClosurePercent } from "./neoApproachTrack";
import type { NeoOrbitTrack } from "./neoTrack";
import { getEventSeverity, SeverityMeter } from "./SeverityMeter";
import { EVENT_TYPE_LABELS, type GlobeAlertEvent } from "./types";
import { timeAgo } from "./useGlobalAlertEvents";

type Props = {
  ev: GlobeAlertEvent;
  enriched: EnrichedStormBriefing | null;
  loading: boolean;
  neoTrack?: NeoOrbitTrack | null;
  neoEnriched?: EnrichedNeoBriefing | null;
  onReturn: () => void;
};

function eventAccentColor(ev: GlobeAlertEvent): string {
  if (ev.type === "hurricane") return hurricaneColorCss(ev.category, ev.severityText);
  return EVENT_TYPE_LABELS[ev.type].color;
}

function HurricaneDock({
  ev,
  enriched,
  loading,
  onReturn,
}: Props) {
  const accent = hurricaneColorCss(ev.category, ev.severityText);
  const wind = enriched?.windKmh;
  const bearing = enriched?.bearingDeg ?? 0;
  const track = enriched?.briefing.track;

  return (
    <div className="ga-focus-dock ga-focus-dock--hurricane" style={{ "--dock-accent": accent } as CSSProperties}>
      <div className="ga-focus-dock__glow" aria-hidden />

      <div className="ga-focus-dock__head">
        <div>
          <span className="ga-focus-dock__live">● LIVE TRACK</span>
          <h2 className="ga-focus-dock__name" style={{ color: accent }}>
            {ev.location}
          </h2>
        </div>
        <button type="button" className="ga-focus-dock__back" onClick={onReturn}>
          ✕
        </button>
      </div>

      {loading ? (
        <p className="ga-focus-dock__loading">מאתר מיקום · מסלול · אוכלוסיה…</p>
      ) : (
        <>
          <div className="ga-focus-dock__hero">
            <div className="ga-focus-dock__wind-ring" style={{ borderColor: accent }}>
              <span className="ga-focus-dock__wind-val">{wind ?? "—"}</span>
              <span className="ga-focus-dock__wind-unit">קמ"ש</span>
              <span className="ga-focus-dock__cat">קט {ev.category ?? "?"}</span>
            </div>

            <div className="ga-focus-dock__compass-wrap">
              <div className="ga-focus-dock__compass">
                <span className="ga-focus-dock__compass-n">N</span>
                <span className="ga-focus-dock__compass-e">E</span>
                <span className="ga-focus-dock__compass-s">S</span>
                <span className="ga-focus-dock__compass-w">W</span>
                <div
                  className="ga-focus-dock__compass-arrow"
                  style={{ transform: `rotate(${bearing}deg)` }}
                  aria-hidden
                />
              </div>
              <span className="ga-focus-dock__bearing">{enriched?.bearingLabel ?? "—"}</span>
              {enriched?.trackSpeedKmh != null ? (
                <span className="ga-focus-dock__track-speed">{enriched.trackSpeedKmh} קמ"ש תנועה</span>
              ) : null}
            </div>
          </div>

          <p className="ga-focus-dock__region">{enriched?.currentRegion ?? ev.location}</p>

          <div className="ga-focus-dock__grid">
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">קואורדינטות</span>
              <span className="ga-focus-dock__cell-val">{enriched?.coordsLabel ?? "—"}</span>
            </div>
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">יעד תחזית</span>
              <span className="ga-focus-dock__cell-val">
                {enriched?.targetRegion?.split(" · ")[0] ?? "בעדכון"}
              </span>
            </div>
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">ETA</span>
              <span className="ga-focus-dock__cell-val">
                {enriched?.etaHours != null && enriched.etaHours > 0
                  ? `~${enriched.etaHours} שע'`
                  : "—"}
              </span>
            </div>
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">מסלול</span>
              <span className="ga-focus-dock__cell-val">
                {track ? `${track.observed.length}+${track.forecast.length} נק'` : "—"}
              </span>
            </div>
          </div>

          {enriched?.populationLabel ? (
            <div className="ga-focus-dock__pop" style={{ borderColor: accent }}>
              <span className="ga-focus-dock__pop-label">אוכלוסיה בסיכון</span>
              <span className="ga-focus-dock__pop-val">{enriched.populationLabel}</span>
            </div>
          ) : null}

          {enriched?.movementLine ? (
            <p className="ga-focus-dock__move">{enriched.movementLine}</p>
          ) : null}
        </>
      )}

      <div className="ga-focus-dock__foot">
        <SeverityMeter ev={ev} color={accent} compact />
        <span className="ga-focus-dock__meta">{timeAgo(ev.time)} · GDACS</span>
      </div>
    </div>
  );
}

function NeoDock({
  ev,
  loading,
  neoTrack,
  neoEnriched,
  onReturn,
}: {
  ev: GlobeAlertEvent;
  loading: boolean;
  neoTrack?: NeoOrbitTrack | null;
  neoEnriched?: EnrichedNeoBriefing | null;
  onReturn: () => void;
}) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, []);
  const live = liveNeoMetrics(ev, neoTrack);
  const caLd = ev.distLd ?? 1;
  const farLd = neoTrack?.points[0]?.distLd ?? live.distLd;
  const closurePct = approachClosurePercent(live.distLd, caLd, farLd);
  const accent =
    live.distLd < 1 ? "#ff6644" : live.distLd < 5 ? "#ffcc44" : EVENT_TYPE_LABELS.neo.color;
  const riskColor =
    neoEnriched?.publicRisk === "critical"
      ? "#ff4444"
      : neoEnriched?.publicRisk === "high"
        ? "#ff8844"
        : neoEnriched?.publicRisk === "moderate"
          ? "#ffcc44"
          : "#66ddff";

  return (
    <div className="ga-focus-dock ga-focus-dock--neo" style={{ "--dock-accent": accent } as CSSProperties}>
      <div className="ga-focus-dock__glow" aria-hidden />
      <div className="ga-focus-dock__scanlines" aria-hidden />

      <div className="ga-focus-dock__head">
        <div>
          <span className="ga-focus-dock__live">● TRACK LOCK · NEO</span>
          <h2 className="ga-focus-dock__name" style={{ color: accent }}>
            {ev.location}
          </h2>
        </div>
        <button type="button" className="ga-focus-dock__back" onClick={onReturn}>
          ✕
        </button>
      </div>

      {loading ? (
        <p className="ga-focus-dock__loading">מאתר מסלול · אזור מעבר · אוכלוסיה…</p>
      ) : (
        <>
          <div className="ga-focus-dock__hero">
            <div className="ga-focus-dock__wind-ring ga-neo-proximity-ring" style={{ borderColor: accent }}>
              <svg className="ga-neo-proximity-ring__svg" viewBox="0 0 100 100" aria-hidden>
                <circle cx="50" cy="50" r="44" className="ga-neo-proximity-ring__track" />
                <circle
                  cx="50"
                  cy="50"
                  r="44"
                  className="ga-neo-proximity-ring__arc"
                  style={{
                    stroke: accent,
                    strokeDasharray: `${closurePct * 2.76} 276`,
                  }}
                />
              </svg>
              <span className="ga-focus-dock__wind-val">{live.distLd.toFixed(2)}</span>
              <span className="ga-focus-dock__wind-unit">LD עכשיו</span>
              <span className="ga-focus-dock__cat">מינ׳ {caLd.toFixed(2)}</span>
            </div>

            <div className="ga-focus-dock__compass-wrap">
              <div className="ga-neo-speed-panel">
                <span className="ga-neo-speed-panel__val">{live.speedKmS.toFixed(1)}</span>
                <span className="ga-neo-speed-panel__unit">km/s</span>
              </div>
              <span className="ga-focus-dock__bearing">מהירות יחסית</span>
              <span className="ga-focus-dock__track-speed">
                {live.diameterKm != null ? `Ø ${live.diameterKm.toFixed(2)} km` : "קוטר בהערכה"}
              </span>
            </div>
          </div>

          <div className="ga-neo-countdown-banner" aria-live="polite">
            <span className="ga-neo-countdown-banner__label">ספירה לקרבה / מעבר</span>
            <time className="ga-neo-countdown-banner__val">
              {formatNeoCountdown(ev.approachTime ?? ev.time)}
            </time>
          </div>

          <p className="ga-focus-dock__region">
            {neoEnriched?.impactRegion ?? `מעל ${ev.lat.toFixed(1)}°, ${ev.lon.toFixed(1)}°`}
          </p>

          <div className="ga-focus-dock__grid">
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">נקודת מעבר</span>
              <span className="ga-focus-dock__cell-val">
                {neoEnriched?.country ?? neoEnriched?.impactRegion.split(" · ").pop() ?? "בעדכון"}
              </span>
            </div>
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">אזור מאוכלס</span>
              <span className="ga-focus-dock__cell-val">
                {neoEnriched
                  ? neoEnriched.isPopulated
                    ? "כן — אזור מיושב"
                    : "לא — ים / מעבר פתוח"
                  : "בעדכון"}
              </span>
            </div>
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">סכנה לציבור</span>
              <span className="ga-focus-dock__cell-val" style={{ color: riskColor }}>
                {neoEnriched?.riskLabel ?? "—"}
              </span>
            </div>
            <div className="ga-focus-dock__cell">
              <span className="ga-focus-dock__cell-label">קואורדינטות</span>
              <span className="ga-focus-dock__cell-val">{neoEnriched?.coordsLabel ?? "—"}</span>
            </div>
          </div>

          {neoEnriched?.populationLabel ? (
            <div className="ga-focus-dock__pop" style={{ borderColor: accent }}>
              <span className="ga-focus-dock__pop-label">אוכלוסיה במדינה</span>
              <span className="ga-focus-dock__pop-val">{neoEnriched.populationLabel}</span>
            </div>
          ) : null}

          {neoEnriched?.riskDetail ? (
            <p className="ga-focus-dock__move">{neoEnriched.riskDetail}</p>
          ) : null}
        </>
      )}

      <div className="ga-focus-dock__foot">
        <SeverityMeter ev={ev} color={accent} compact />
        <span className="ga-focus-dock__meta">NASA JPL · מסלול חי</span>
      </div>
    </div>
  );
}

function GenericDock({ ev, onReturn }: { ev: GlobeAlertEvent; onReturn: () => void }) {
  const ti = EVENT_TYPE_LABELS[ev.type];
  const accent = eventAccentColor(ev);
  const sev = getEventSeverity(ev);

  return (
    <div
      className={`ga-focus-dock ga-focus-dock--generic is-sev-${sev.tier}`}
      style={{ "--dock-accent": accent, borderColor: accent } as CSSProperties}
    >
      <div className="ga-focus-dock__head">
        <div>
          <span className="ga-focus-dock__badge" style={{ color: accent, borderColor: accent }}>
            {ti.label}
          </span>
          <h2 className="ga-focus-dock__name" style={{ color: accent }}>
            {ev.location}
          </h2>
        </div>
        <button type="button" className="ga-focus-dock__back" onClick={onReturn}>
          ✕
        </button>
      </div>

      {ev.severityText ? <p className="ga-focus-dock__detail">{ev.severityText}</p> : null}

      <div className="ga-focus-dock__grid">
        {ev.magnitude != null ? (
          <div className="ga-focus-dock__cell">
            <span className="ga-focus-dock__cell-label">מגניטודה</span>
            <span className="ga-focus-dock__cell-val" style={{ color: accent }}>
              M{ev.magnitude.toFixed(1)}
            </span>
          </div>
        ) : null}
        {ev.depth != null ? (
          <div className="ga-focus-dock__cell">
            <span className="ga-focus-dock__cell-label">עומק</span>
            <span className="ga-focus-dock__cell-val">{ev.depth.toFixed(1)} ק"מ</span>
          </div>
        ) : null}
        {ev.category != null ? (
          <div className="ga-focus-dock__cell">
            <span className="ga-focus-dock__cell-label">קטגוריה</span>
            <span className="ga-focus-dock__cell-val">{ev.category}</span>
          </div>
        ) : null}
        {ev.alertLevel ? (
          <div className="ga-focus-dock__cell">
            <span className="ga-focus-dock__cell-label">רמת התרעה</span>
            <span className="ga-focus-dock__cell-val">{ev.alertLevel}</span>
          </div>
        ) : null}
      </div>

      <div className="ga-focus-dock__foot">
        <SeverityMeter ev={ev} color={accent} compact />
        <span className="ga-focus-dock__meta">{timeAgo(ev.time)} · {ev.source.toUpperCase()}</span>
      </div>
    </div>
  );
}

export function EventFocusDock({ ev, enriched, loading, neoTrack, neoEnriched, onReturn }: Props) {
  if (ev.type === "hurricane") {
    return <HurricaneDock ev={ev} enriched={enriched} loading={loading} onReturn={onReturn} />;
  }
  if (ev.type === "neo") {
    return (
      <NeoDock
        ev={ev}
        loading={loading}
        neoTrack={neoTrack}
        neoEnriched={neoEnriched}
        onReturn={onReturn}
      />
    );
  }
  return <GenericDock ev={ev} onReturn={onReturn} />;
}
