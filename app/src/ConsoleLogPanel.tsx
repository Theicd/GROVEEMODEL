import { useEffect, useMemo, useState } from "react";
import {
  clearConsoleLogs,
  formatConsoleLogs,
  getConsoleLogs,
  subscribeConsoleLogs,
  type ConsoleLogEntry,
  type ConsoleLogLevel,
} from "./consoleLogStore";

const LEVEL_TABS: { id: "all" | ConsoleLogLevel; label: string }[] = [
  { id: "all", label: "הכל" },
  { id: "error", label: "שגיאות" },
  { id: "warn", label: "אזהרות" },
  { id: "info", label: "מידע" },
];

function levelClass(level: ConsoleLogLevel): string {
  if (level === "error") return "clog-line--error";
  if (level === "warn") return "clog-line--warn";
  if (level === "info") return "clog-line--info";
  return "clog-line--log";
}

export function ConsoleLogPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [entries, setEntries] = useState<ConsoleLogEntry[]>([]);
  const [filter, setFilter] = useState<"all" | ConsoleLogLevel>("all");
  const [copyState, setCopyState] = useState<"idle" | "ok" | "fail">("idle");

  useEffect(() => {
    if (!open) return;
    setEntries(getConsoleLogs());
    const unsub = subscribeConsoleLogs(() => setEntries(getConsoleLogs()));
    return unsub;
  }, [open]);

  const visible = useMemo(() => {
    if (filter === "all") return entries;
    if (filter === "warn") return entries.filter((e) => e.level === "warn" || e.level === "error");
    return entries.filter((e) => e.level === filter);
  }, [entries, filter]);

  const errorCount = useMemo(() => entries.filter((e) => e.level === "error").length, [entries]);

  if (!open) return null;

  const copyAll = async () => {
    const text = formatConsoleLogs(visible.length ? visible : entries);
    try {
      await navigator.clipboard.writeText(text);
      setCopyState("ok");
    } catch {
      // Clipboard API often blocked on mobile / non-secure contexts — fall back.
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        setCopyState("ok");
      } catch {
        setCopyState("fail");
      }
    }
    window.setTimeout(() => setCopyState("idle"), 2200);
  };

  const copyLabel =
    copyState === "ok" ? "הועתק!" : copyState === "fail" ? "העתקה נכשלה" : "העתק הכל";

  return (
    <div
      className="activity-overlay modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="clog-panel-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="activity-panel modal-box clog-panel">
        <header className="activity-panel-head">
          <div>
            <h2 id="clog-panel-title">לוג הקונסולה</h2>
            <p className="activity-panel-sub">
              שגיאות ופלט הרקע של הדפדפן — לבדיקה ודיבוג במובייל
            </p>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </header>

        <div className="activity-panel-toolbar clog-toolbar">
          <div className="clog-tabs">
            {LEVEL_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`clog-tab${filter === tab.id ? " clog-tab--active" : ""}`}
                onClick={() => setFilter(tab.id)}
              >
                {tab.label}
                {tab.id === "error" && errorCount > 0 ? ` (${errorCount})` : ""}
              </button>
            ))}
          </div>
          <div className="activity-panel-toolbar-actions">
            <button
              type="button"
              className={`activity-copy-all-btn ${copyState === "ok" ? "activity-copy-all-btn--ok" : ""}`}
              onClick={() => void copyAll()}
              disabled={!entries.length}
              title="העתק את הלוג כטקסט"
            >
              {copyLabel}
            </button>
            <button
              type="button"
              className="activity-clear-btn"
              onClick={() => clearConsoleLogs()}
              disabled={!entries.length}
            >
              נקה
            </button>
          </div>
        </div>

        <div className="activity-list clog-list" dir="ltr">
          {!visible.length ? (
            <p className="activity-empty">אין רשומות עדיין.</p>
          ) : (
            visible.map((e) => (
              <div key={e.id} className={`clog-line ${levelClass(e.level)} clog-line--${e.source}`}>
                <span className="clog-time">{new Date(e.ts).toISOString().slice(11, 23)}</span>
                <span className="clog-level">{e.level.toUpperCase()}</span>
                {e.source === "worker" ? <span className="clog-src">worker</span> : null}
                <span className="clog-text">{e.text}</span>
              </div>
            ))
          )}
        </div>

        <footer className="activity-panel-foot">
          <span className="activity-count">{entries.length} רשומות</span>
          <button type="button" className="settings-btn-save" onClick={onClose}>
            סגור
          </button>
        </footer>
      </div>
    </div>
  );
}
