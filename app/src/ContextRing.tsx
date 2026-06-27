import { useEffect, useRef, useState } from "react";
import { approxTokensFromChars, formatTokenCount } from "./contextUsageEstimate";

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
  { key: "user", labelHe: "טיוטה / הודעה נוכחית", color: "#f59e0b" },
  { key: "images", labelHe: "תמונות וקבצים", color: "#ec4899" },
];

/**
 * Circular context meter (ChatGPT / Cursor style): ring fills as context is used.
 * Click opens a breakdown popover with token counts and segment bars.
 */
export function ContextRing({ usage }: Props) {
  const [open, setOpen] = useState(false);
  const [mobile, setMobile] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(max-width: 768px)");
    const sync = () => setMobile(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  const freePercent = Math.max(0, Math.min(100, Math.round(usage.percent)));
  const usedPercent = 100 - freePercent;
  const tone = freePercent > 50 ? "ok" : freePercent > 20 ? "warn" : "danger";

  const usedTokens = approxTokensFromChars(usage.usedChars);
  const totalTokens = approxTokensFromChars(usage.totalBudget);
  const usedTotalChars = Object.values(usage.breakdown).reduce((a, b) => a + b, 0) || 1;

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
  const dash = (usedPercent / 100) * CIRC;

  return (
    <div className="context-ring-root" ref={rootRef}>
      <button
        type="button"
        className={`context-ring context-ring--${tone}`}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`הקשר שיחה: ${usedPercent}% בשימוש, ${freePercent}% פנוי`}
        title={`הקשר שיחה — ${formatTokenCount(usedTokens)} / ${formatTokenCount(totalTokens)} טוקנים`}
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
        <span className="context-ring-pct">{usedPercent}%</span>
      </button>

      {open ? (
        <div
          className={`context-popover${mobile ? " context-popover--mobile" : ""}`}
          role="dialog"
          aria-label="פירוט הקשר שיחה"
        >
          <div className="context-popover-head">
            <div>
              <strong className="context-popover-title">חלון הקשר</strong>
              <span className="context-popover-sub">Context window</span>
            </div>
            <span className={`context-popover-badge context-popover-badge--${tone}`}>
              {freePercent}% פנוי
            </span>
          </div>

          <div className="context-popover-hero">
            <div className="context-popover-hero-main">
              <span className="context-popover-hero-used">{formatTokenCount(usedTokens)}</span>
              <span className="context-popover-hero-sep">/</span>
              <span className="context-popover-hero-total">{formatTokenCount(totalTokens)}</span>
              <span className="context-popover-hero-unit">tokens</span>
            </div>
            <div className="context-popover-hero-meta">
              <span>{usedPercent}% בשימוש</span>
              <span aria-hidden="true">·</span>
              <span>{freePercent}% זמין</span>
            </div>
          </div>

          <div
            className="context-popover-meter"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={usedPercent}
            aria-label={`${usedPercent}% מההקשר בשימוש`}
          >
            <span
              className={`context-popover-meter-fill context-popover-meter-fill--${tone}`}
              style={{ width: `${Math.max(usedPercent, usedTokens > 0 ? 2 : 0)}%` }}
            />
          </div>

          <ul className="context-popover-rows">
            {SEGMENTS.map((seg) => {
              const chars = usage.breakdown[seg.key];
              if (chars <= 0) return null;
              const tokens = approxTokensFromChars(chars);
              const share = Math.round((chars / usedTotalChars) * 100);
              return (
                <li key={seg.key} className="context-popover-row">
                  <div className="context-popover-row-top">
                    <span className="context-popover-dot" style={{ background: seg.color }} />
                    <span className="context-popover-row-label">{seg.labelHe}</span>
                    <span className="context-popover-row-val">{formatTokenCount(tokens)}</span>
                  </div>
                  <div className="context-popover-row-bar" aria-hidden="true">
                    <span
                      className="context-popover-row-bar-fill"
                      style={{ width: `${Math.max(share, 2)}%`, background: seg.color }}
                    />
                  </div>
                  <span className="context-popover-row-share">{share}% מהשימוש</span>
                </li>
              );
            })}
          </ul>

          <div className="context-popover-foot">
            <span className="context-popover-profile">פרופיל: {usage.profileLabel}</span>
            {freePercent <= 20 ? (
              <p className="context-popover-warn">
                ההקשר כמעט מלא — התשובות עלולות להיות קצרות. מומלץ לפתוח שיחה חדשה.
              </p>
            ) : freePercent <= 50 ? (
              <p className="context-popover-hint">שיחה ארוכה — שקול לסכם או להתחיל שיחה חדשה.</p>
            ) : (
              <p className="context-popover-hint">הערכה לפי היסטוריה, הנחיות מערכת והטיוטה הנוכחית.</p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
