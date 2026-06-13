import { useEffect, useRef, useState } from "react";

export type ContextUsage = {
  /** Remaining context, 0–100. */
  percent: number;
  usedChars: number;
  totalBudget: number;
  profileLabel: string;
  breakdown: { history: number; web: number; system: number; user: number; images: number };
};

type Props = {
  usage: ContextUsage;
};

const SEGMENTS: Array<{ key: keyof ContextUsage["breakdown"]; labelHe: string; color: string }> = [
  { key: "history", labelHe: "היסטוריית שיחה", color: "#10a37f" },
  { key: "web", labelHe: "תוצאות חיפוש", color: "#00b8d9" },
  { key: "system", labelHe: "הנחיות מערכת", color: "#8b5cf6" },
  { key: "user", labelHe: "ההודעה הנוכחית", color: "#f59e0b" },
  { key: "images", labelHe: "תמונות", color: "#ec4899" },
];

const approxTokens = (chars: number) => Math.max(0, Math.round(chars / 4));

const formatTokens = (n: number) =>
  n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n);

/**
 * Circular context meter (Cursor/ChatGPT style): a small ring near the composer
 * showing remaining context %. Click opens a breakdown popover.
 */
export function ContextRing({ usage }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const p = Math.max(0, Math.min(100, Math.round(usage.percent)));
  const tone = p > 50 ? "ok" : p > 20 ? "warn" : "danger";

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const R = 8;
  const CIRC = 2 * Math.PI * R;
  const dash = (p / 100) * CIRC;

  const usedTotal = Object.values(usage.breakdown).reduce((a, b) => a + b, 0) || 1;

  return (
    <div className="context-ring-root" ref={rootRef} dir="rtl">
      <button
        type="button"
        className={`context-ring context-ring--${tone}`}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`הקשר שיחה: ${p}% פנוי`}
        title={`הקשר שיחה: ${p}% פנוי — לחץ לפירוט`}
      >
        <svg viewBox="0 0 22 22" width="22" height="22" aria-hidden="true">
          <circle className="context-ring-track" cx="11" cy="11" r={R} />
          <circle
            className="context-ring-fill"
            cx="11"
            cy="11"
            r={R}
            strokeDasharray={`${dash} ${CIRC - dash}`}
            transform="rotate(-90 11 11)"
          />
        </svg>
        <span className="context-ring-pct">{p}%</span>
      </button>

      {open ? (
        <div className="context-popover" role="dialog" aria-label="פירוט הקשר שיחה">
          <div className="context-popover-head">
            <strong>הקשר שיחה</strong>
            <span className={`context-popover-pct context-popover-pct--${tone}`}>{p}% פנוי</span>
          </div>
          <div className="context-popover-tokens">
            ~{formatTokens(approxTokens(usage.usedChars))} מתוך ~{formatTokens(approxTokens(usage.totalBudget))} טוקנים בשימוש
          </div>

          <div className="context-popover-stack" aria-hidden="true">
            {SEGMENTS.map((seg) => {
              const v = usage.breakdown[seg.key];
              if (v <= 0) return null;
              return (
                <span
                  key={seg.key}
                  className="context-popover-stack-seg"
                  style={{
                    width: `${Math.max(2, (v / usedTotal) * 100)}%`,
                    background: seg.color,
                  }}
                />
              );
            })}
          </div>

          <ul className="context-popover-rows">
            {SEGMENTS.map((seg) => {
              const v = usage.breakdown[seg.key];
              if (v <= 0) return null;
              return (
                <li key={seg.key} className="context-popover-row">
                  <span className="context-popover-dot" style={{ background: seg.color }} />
                  <span className="context-popover-row-label">{seg.labelHe}</span>
                  <span className="context-popover-row-val">~{formatTokens(approxTokens(v))}</span>
                </li>
              );
            })}
          </ul>

          <div className="context-popover-foot">
            פרופיל חומרה: {usage.profileLabel}
            {p <= 20 ? (
              <span className="context-popover-warn">ההקשר כמעט מלא — מומלץ לפתוח שיחה חדשה</span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
