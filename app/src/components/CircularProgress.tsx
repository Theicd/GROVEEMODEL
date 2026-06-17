import { formatDownloadPercent } from "../introProgressFormat";

type CircularProgressProps = {
  percent: number;
  size?: number;
  label?: string;
  indeterminate?: boolean;
};

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function CircularProgress({
  percent,
  size = 112,
  label,
  indeterminate = false,
}: CircularProgressProps) {
  const clamped = Math.min(100, Math.max(0, percent));
  const offset = CIRCUMFERENCE * (1 - clamped / 100);

  return (
    <div
      className={`hal-ring${indeterminate ? " hal-ring--indeterminate" : ""}`}
      style={{ width: size, height: size }}
      data-testid="download-ring"
      aria-valuenow={clamped}
      aria-valuemin={0}
      aria-valuemax={100}
      role="progressbar"
    >
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <circle className="hal-ring__track" cx="60" cy="60" r={RADIUS} />
        <circle
          className="hal-ring__fill"
          cx="60"
          cy="60"
          r={RADIUS}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={indeterminate ? CIRCUMFERENCE * 0.72 : offset}
        />
      </svg>
      <div className="hal-ring__center" dir="ltr">
        <span className="hal-ring__pct">
          {indeterminate ? "…" : `${formatDownloadPercent(clamped)}%`}
        </span>
        {label ? <span className="hal-ring__label">{label}</span> : null}
      </div>
    </div>
  );
}
