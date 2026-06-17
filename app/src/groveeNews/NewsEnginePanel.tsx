import { useCallback, useEffect, useState } from "react";
import { RSS_POLL_INTERVAL_MS } from "./engine/feeds/feedRegistry";
import { forcePollAllFeeds, refreshEngineStats } from "./engine/engine/pipeline";
import { isAiDeepReadEnabled, subscribeAiDeepRead } from "./engine/settings/aiMode";
import type { EngineStatus } from "./engine/types";
import { AiDeepReadPanel } from "./AiDeepReadPanel";
import { isNewsEngineBusy, newsFeedScanPercent, useNewsEngineStatus } from "./useNewsEngineStatus";
import "./newsEnginePanel.css";

type EngineTab = "status" | "log";

function fmtTime(ts: number): string {
  if (!ts) return "—";
  return new Date(ts).toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtNum(n: number): string {
  return n.toLocaleString("he-IL");
}

function nextPollShort(lastPollAt: number, phase: EngineStatus["phase"]): string {
  if (phase === "polling") return "סורק עכשיו";
  if (!lastPollAt) return "לחץ רענון מקורות";
  const remain = lastPollAt + RSS_POLL_INTERVAL_MS - Date.now();
  if (remain <= 0) return "בקרוב";
  return `~${Math.ceil(remain / 60_000)} דק׳`;
}

function phaseLabelHe(status: EngineStatus, aiOn: boolean): string {
  if (!aiOn && status.phase === "ready") {
    return "מנוע פעיל — אוסף כותרות RSS (חקירה עמוקה כבויה)";
  }
  switch (status.phase) {
    case "polling":
      return status.message || `סורק מקורות ${status.feedsOk + status.feedsFailed}/${status.feedsTotal}…`;
    case "extracting":
      return "שולף עמודי כתבות מלאים…";
    case "summarizing":
      return "Qwen מסכם כתבות…";
    case "indexing":
      return "מעדכן אינדקס חיפוש…";
    case "ready":
      return status.message || "מוכן — אוסף ברקע כל 5 דקות";
    case "error":
      return status.message || "שגיאה במנוע";
    default:
      return status.message || "ממתין לסריקה ראשונה…";
  }
}

function kindIcon(kind: string): string {
  switch (kind) {
    case "rss":
      return "◉";
    case "extract":
      return "↓";
    case "summarize":
      return "◎";
    case "index":
      return "⌗";
    case "search":
      return "⌕";
    case "model":
      return "◈";
    case "connector":
      return "⎇";
    case "error":
      return "✗";
    default:
      return "·";
  }
}

function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="gn-engine-stat">
      <span className="gn-engine-stat__label">{label}</span>
      <strong className="gn-engine-stat__value">{value}</strong>
      {hint ? <span className="gn-engine-stat__hint">{hint}</span> : null}
    </div>
  );
}

export function NewsEnginePanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { status } = useNewsEngineStatus();
  const [aiOn, setAiOn] = useState(isAiDeepReadEnabled);
  const [tab, setTab] = useState<EngineTab>("status");
  const [refreshing, setRefreshing] = useState(false);

  const busy = isNewsEngineBusy(status);
  const scanPct = newsFeedScanPercent(status);
  const headlines = status.library?.rssHeadlines ?? status.rssHeadlines;
  const articles = status.library?.articlesIndexed ?? status.articlesIndexed;
  const pending = status.library?.pendingArticles ?? status.pendingArticles;
  const summarized = status.library?.summarizedByModel ?? status.summarizedByModel;

  useEffect(() => subscribeAiDeepRead(() => setAiOn(isAiDeepReadEnabled())), []);

  const reloadStats = useCallback(() => {
    void refreshEngineStats();
  }, []);

  useEffect(() => {
    if (!open) return;
    reloadStats();
    const timer = window.setInterval(reloadStats, 8_000);
    return () => window.clearInterval(timer);
  }, [open, reloadStats]);

  const onRefreshFeeds = async () => {
    setRefreshing(true);
    try {
      await forcePollAllFeeds();
      await refreshEngineStats();
    } finally {
      setRefreshing(false);
    }
  };

  if (!open) return null;

  const phaseClass = busy ? "polling" : headlines > 0 ? "ready" : "idle";

  return (
    <div
      className="activity-overlay modal news-engine-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="news-engine-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="activity-panel modal-box news-engine-panel">
        <header className="activity-panel-head">
          <div>
            <h2 id="news-engine-title">מנוע חדשות ברקע</h2>
            <p className="activity-panel-sub">RSS · אינדקס · חקירה עמוקה עם Qwen</p>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </header>

        <div className="activity-panel-toolbar news-engine-toolbar">
          <span className={`news-engine-live ${busy ? "news-engine-live--on" : ""}`}>
            {busy ? "● פעיל עכשיו" : headlines > 0 ? `${fmtNum(headlines)} כותרות באוסף` : "ממתין לנתונים"}
          </span>
          <button
            type="button"
            className="gn-engine-panel__refresh"
            onClick={() => void onRefreshFeeds()}
            disabled={refreshing || status.phase === "polling"}
          >
            {status.phase === "polling" || refreshing ? "סורק…" : "רענון מקורות"}
          </button>
        </div>

        <nav className="gn-engine-tabs news-engine-tabs" aria-label="לשוניות מנוע חדשות">
          <button
            type="button"
            className={`gn-engine-tabs__btn${tab === "status" ? " gn-engine-tabs__btn--active" : ""}`}
            onClick={() => setTab("status")}
          >
            סטטוס
          </button>
          <button
            type="button"
            className={`gn-engine-tabs__btn${tab === "log" ? " gn-engine-tabs__btn--active" : ""}`}
            onClick={() => setTab("log")}
          >
            לוג
            {status.activityLog.length ? (
              <span className="gn-engine-tabs__badge">{Math.min(status.activityLog.length, 99)}</span>
            ) : null}
          </button>
        </nav>

        <div className="news-engine-body">
          {tab === "status" ? (
            <>
              <p className={`gn-engine-phase gn-engine-phase--${phaseClass}`}>{phaseLabelHe(status, aiOn)}</p>

              {busy && status.phase === "polling" && status.feedsTotal > 0 ? (
                <div className="news-engine-progress" role="progressbar" aria-valuenow={scanPct} aria-valuemin={0} aria-valuemax={100}>
                  <div className="news-engine-progress__track">
                    <div className="news-engine-progress__fill" style={{ width: `${scanPct}%` }} />
                  </div>
                  <span className="news-engine-progress__label">
                    {scanPct}% · {status.feedsOk + status.feedsFailed}/{status.feedsTotal} מקורות
                  </span>
                </div>
              ) : null}

              <dl className="gn-db-summary">
                <div>
                  <dt>סריקה אחרונה</dt>
                  <dd>{status.lastPollAt ? fmtTime(status.lastPollAt) : busy ? "בתהליך…" : "מעולם לא"}</dd>
                </div>
                <div>
                  <dt>סריקה הבאה</dt>
                  <dd>{nextPollShort(status.lastPollAt, status.phase)}</dd>
                </div>
                <div>
                  <dt>מקורות RSS</dt>
                  <dd>
                    {busy && status.feedsOk + status.feedsFailed === 0
                      ? `מתחיל 0/${status.feedsTotal}…`
                      : `${status.feedsOk}/${status.feedsTotal} תקין · ${status.feedsFailed} נכשל`}
                  </dd>
                </div>
              </dl>

              <div className="gn-engine-stat-grid gn-engine-stat-grid--compact">
                <StatCard label="כותרות RSS" value={fmtNum(headlines)} hint="נשמר ב-IndexedDB" />
                <StatCard label="בחיפוש" value={fmtNum(articles)} hint="FlexSearch" />
                <StatCard label="בתור" value={fmtNum(pending)} hint="ממתין לעיבוד" />
                <StatCard
                  label="סוכמו ב-Qwen"
                  value={fmtNum(summarized)}
                  hint={aiOn ? "כתבות מלאות" : "הפעל חקירה עמוקה"}
                />
              </div>

              <AiDeepReadPanel />
            </>
          ) : (
            <ul className="gn-activity-log gn-activity-log--panel">
              {status.activityLog.length === 0 ? (
                <li className="gn-activity-log__empty">אין פעילות עדיין — המנוע יתחיל לאסוף אחרי טעינת האפליקציה.</li>
              ) : (
                status.activityLog.map((e) => (
                  <li key={`${e.ts}-${e.message}`} className={`gn-activity-log__item gn-activity-log__item--${e.kind}`}>
                    <span className="gn-activity-log__time">{fmtTime(e.ts)}</span>
                    <span className="gn-activity-log__icon">{kindIcon(e.kind)}</span>
                    <span className="gn-activity-log__msg">{e.message}</span>
                  </li>
                ))
              )}
            </ul>
          )}
        </div>

        {status.lastSummary ? (
          <footer className="activity-panel-foot news-engine-foot">
            <span className="news-engine-foot__label">
              סיכום אחרון {status.lastSummary.byModel ? "(Qwen)" : "(RSS)"}:
            </span>
            <span className="news-engine-foot__title">{status.lastSummary.title}</span>
          </footer>
        ) : null}
      </div>
    </div>
  );
}

/** Compact badge for sidebar — headline count or live pulse. */
export function NewsEngineRailHint({
  status,
  countClassName = "sb-rail-count",
}: {
  status: EngineStatus;
  countClassName?: string;
}) {
  const busy = isNewsEngineBusy(status);
  const headlines = status.library?.rssHeadlines ?? status.rssHeadlines;
  if (busy) {
    return (
      <span className={`${countClassName} sb-rail-count--live`} aria-hidden="true">
        ●
      </span>
    );
  }
  if (headlines > 0) {
    return (
      <span className={countClassName} aria-hidden="true">
        {headlines > 999 ? "999+" : headlines}
      </span>
    );
  }
  return null;
}
