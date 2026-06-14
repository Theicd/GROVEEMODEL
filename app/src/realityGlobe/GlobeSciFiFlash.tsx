import type { IntelFlashAlert } from "./intelFeed";

type Props = {
  alert: IntelFlashAlert | null;
  onDismiss: () => void;
};

function formatEventTime(iso?: string): string | null {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleString("he-IL", {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return null;
  }
}

export function GlobeSciFiFlash({ alert, onDismiss }: Props) {
  if (!alert) return null;

  const critical = alert.severity >= 5;
  const when = formatEventTime(alert.eventTime);

  return (
    <div
      className={`globe-scifi-flash${critical ? " globe-scifi-flash--critical" : ""}`}
      role="alert"
      onClick={onDismiss}
    >
      <div className="globe-scifi-flash-vignette" aria-hidden="true" />
      <div className="globe-scifi-flash-scanlines" aria-hidden="true" />
      <div className="globe-scifi-flash-box">
        <div className="globe-scifi-flash-corner globe-scifi-flash-corner--tl" />
        <div className="globe-scifi-flash-corner globe-scifi-flash-corner--tr" />
        <div className="globe-scifi-flash-corner globe-scifi-flash-corner--bl" />
        <div className="globe-scifi-flash-corner globe-scifi-flash-corner--br" />
        <span className="globe-scifi-flash-tag">
          {alert.category} · S{alert.severity}
        </span>
        <h3 className="globe-scifi-flash-title">{alert.title}</h3>
        <p className="globe-scifi-flash-body">{alert.body}</p>
        <dl className="globe-scifi-flash-meta">
          {alert.place ? (
            <>
              <dt>מיקום</dt>
              <dd>{alert.place}</dd>
            </>
          ) : null}
          {alert.lat != null && alert.lon != null ? (
            <>
              <dt>קואורדינטות</dt>
              <dd>
                {alert.lat.toFixed(2)}°, {alert.lon.toFixed(2)}°
              </dd>
            </>
          ) : null}
          {alert.magnitude != null ? (
            <>
              <dt>עוצמה</dt>
              <dd>M{alert.magnitude.toFixed(1)}</dd>
            </>
          ) : null}
          {alert.depth != null && Number.isFinite(alert.depth) ? (
            <>
              <dt>עומק</dt>
              <dd>{Math.round(alert.depth)} km</dd>
            </>
          ) : null}
          {when ? (
            <>
              <dt>זמן</dt>
              <dd>{when}</dd>
            </>
          ) : null}
          {alert.source ? (
            <>
              <dt>מקור</dt>
              <dd>{alert.source}</dd>
            </>
          ) : null}
        </dl>
        {alert.recommendedAction ? (
          <p className="globe-scifi-flash-action">→ {alert.recommendedAction}</p>
        ) : null}
        <p className="globe-scifi-flash-hint">לחץ לסגירה</p>
      </div>
    </div>
  );
}
