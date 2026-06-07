import { useState } from "react";
import {
  formatActivityLogForCopy,
  formatActivityTime,
  directionLabel,
  type ModelActivityEntry,
} from "./modelActivityLog";

export function ModelActivityPanel({
  open,
  onClose,
  entries,
  onClear,
}: {
  open: boolean;
  onClose: () => void;
  entries: ModelActivityEntry[];
  onClear: () => void;
}) {
  const [copyState, setCopyState] = useState<"idle" | "ok" | "fail">("idle");

  if (!open) return null;

  const copyFullLog = async () => {
    if (!entries.length) return;
    const text = formatActivityLogForCopy(entries);
    try {
      await navigator.clipboard.writeText(text);
      setCopyState("ok");
      window.setTimeout(() => setCopyState("idle"), 2000);
    } catch {
      setCopyState("fail");
      window.setTimeout(() => setCopyState("idle"), 2500);
    }
  };

  const copyLabel =
    copyState === "ok" ? "הועתק!" : copyState === "fail" ? "העתקה נכשלה" : "העתק לוג";

  return (
    <div
      className="activity-overlay modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="activity-panel-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="activity-panel modal-box">
        <header className="activity-panel-head">
          <div>
            <h2 id="activity-panel-title">פעילות המודל</h2>
            <p className="activity-panel-sub">כל הבקשות, ההנחיות והתשובות — לבדיקה ודיבוג</p>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </header>

        <div className="activity-panel-toolbar">
          <span className="activity-count">{entries.length} רשומות</span>
          <div className="activity-panel-toolbar-actions">
            <button
              type="button"
              className={`activity-copy-all-btn ${copyState === "ok" ? "activity-copy-all-btn--ok" : ""}`}
              onClick={() => void copyFullLog()}
              disabled={!entries.length}
              title="העתק את כל הלוג כטקסט מסודר"
            >
              {copyLabel}
            </button>
            <button type="button" className="activity-clear-btn" onClick={onClear} disabled={!entries.length}>
              נקה לוג
            </button>
          </div>
        </div>

        <div className="activity-list" dir="ltr">
          {!entries.length ? (
            <p className="activity-empty">אין פעילות עדיין. שלח הודעה או הפעל Camera Mode.</p>
          ) : (
            entries.map((entry) => (
              <article key={entry.id} className={`activity-item activity-item--${entry.direction}`}>
                <div className="activity-item-head">
                  <span className={`activity-dir activity-dir--${entry.direction}`}>
                    {directionLabel(entry.direction)}
                  </span>
                  <span className="activity-kind">{entry.kind}</span>
                  <time className="activity-time">{formatActivityTime(entry.ts)}</time>
                </div>
                <h3 className="activity-title">{entry.title}</h3>
                {entry.meta && Object.keys(entry.meta).length > 0 ? (
                  <dl className="activity-meta">
                    {Object.entries(entry.meta).map(([k, v]) => (
                      <div key={k} className="activity-meta-row">
                        <dt>{k}</dt>
                        <dd>{String(v)}</dd>
                      </div>
                    ))}
                  </dl>
                ) : null}
                <pre className="activity-detail">{entry.detail}</pre>
              </article>
            ))
          )}
        </div>

        <footer className="activity-panel-foot">
          <button type="button" className="settings-btn-save" onClick={onClose}>
            סגור
          </button>
        </footer>
      </div>
    </div>
  );
}
