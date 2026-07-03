import type { CSSProperties } from "react";
import type { GlobeAlertEvent } from "./types";
import { getEventSeverity } from "./severityScore";

type Props = {
  ev: GlobeAlertEvent;
  color: string;
  compact?: boolean;
};

export function SeverityMeter({ ev, color, compact = false }: Props) {
  const sev = getEventSeverity(ev);
  return (
    <div
      className={`ga-severity ga-severity--${sev.tier}${compact ? " ga-severity--compact" : ""}`}
      style={{ "--sev-color": color } as CSSProperties}
      title={`חומרה: ${sev.label}`}
      aria-label={`רמת חומרה ${sev.label}`}
    >
      <div className="ga-severity__bars" aria-hidden>
        {[1, 2, 3, 4].map((i) => (
          <span key={i} className={`ga-severity__bar${i <= sev.bars ? " is-on" : ""}`} />
        ))}
      </div>
      <span className="ga-severity__label">{sev.label}</span>
    </div>
  );
}

export { getEventSeverity };
