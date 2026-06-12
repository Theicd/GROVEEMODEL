import type { IntelFlashAlert } from "./intelFeed";

type Props = {
  alert: IntelFlashAlert | null;
  onDismiss: () => void;
};

export function GlobeSciFiFlash({ alert, onDismiss }: Props) {
  if (!alert) return null;

  const critical = alert.severity >= 5;

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
        <span className="globe-scifi-flash-tag">{alert.category}</span>
        <h3 className="globe-scifi-flash-title">{alert.title}</h3>
        <p className="globe-scifi-flash-body">{alert.body}</p>
      </div>
    </div>
  );
}
