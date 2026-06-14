import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { directionLabel, formatActivityTime, type ModelActivityEntry } from "./modelActivityLog";
import { qaChatBridge } from "./qaChatBridge";
import {
  autoGradeResult,
  buildPresentationQaReport,
  clearQaResults,
  effectiveStatus,
  loadQaResults,
  QA_CODE_VERSION,
  QA_STATUS_ICON,
  saveQaResults,
  type QaManualStatus,
  type QaRecordedResult,
} from "./presentationQaGrade";
import { SearchProgressPanel } from "./SearchProgressPanel";
import { USER_PRESENTATION_QUERIES, type UserPresentationQuery } from "./userPresentationQueries";
import type { SearchBrief, SearchSourceResult } from "./webSearch/types";

const GROUP_LABEL: Record<UserPresentationQuery["group"], string> = {
  basic: "יכולות בסיס",
  cross: "הצלבת מקורות",
  natural: "שאלות טבעיות",
};

export type QaStreamingSearch = {
  sources: SearchSourceResult[];
  summary: string;
  query?: string;
  brief?: SearchBrief;
  active?: boolean;
} | null;

function livePhase(opts: {
  running: boolean;
  isGenerating: boolean;
  search: QaStreamingSearch;
  assistantBuffer: string;
}): { label: string; step: number } {
  if (!opts.running) return { label: "", step: 0 };
  if (opts.search?.active) return { label: "מחפש ברשת", step: 2 };
  if (opts.isGenerating && !opts.assistantBuffer.trim()) return { label: "שולח ל-Gemma", step: 3 };
  if (opts.isGenerating && opts.assistantBuffer.trim()) return { label: "Gemma כותב", step: 4 };
  if (!opts.isGenerating && !opts.search?.active) return { label: "תשובת canned", step: 3 };
  return { label: "מתחיל", step: 1 };
}

const STEPS = ["▶", "חיפוש", "שליחה", "כתיבה"];

export function PresentationQaPanel({
  open,
  onClose,
  modelReady,
  isGenerating,
  activityLog,
  assistantBuffer,
  streamingSearch,
}: {
  open: boolean;
  onClose: () => void;
  modelReady: boolean;
  isGenerating: boolean;
  activityLog: ModelActivityEntry[];
  assistantBuffer: string;
  streamingSearch: QaStreamingSearch;
}) {
  const [results, setResults] = useState<Record<string, QaRecordedResult>>(loadQaResults);
  const [selectedId, setSelectedId] = useState<string | null>(USER_PRESENTATION_QUERIES[0]?.id ?? null);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [activitySince, setActivitySince] = useState<number>(0);
  const [forceLlm, setForceLlm] = useState(false);
  const [filterGroup, setFilterGroup] = useState<"all" | UserPresentationQuery["group"]>("all");
  const [copyState, setCopyState] = useState<"idle" | "ok" | "fail">("idle");
  const [error, setError] = useState<string | null>(null);
  const [autoRunning, setAutoRunning] = useState(false);
  const [autoProgress, setAutoProgress] = useState<{ current: number; total: number; id: string } | null>(
    null,
  );
  const [skipTested, setSkipTested] = useState(false);
  const autoRunRef = useRef(false);
  const forceLlmRef = useRef(forceLlm);
  forceLlmRef.current = forceLlm;

  useEffect(() => {
    saveQaResults(results);
  }, [results]);

  useEffect(() => {
    if (!runningId || !runStartedAt) {
      setElapsedSec(0);
      return;
    }
    const tick = () => setElapsedSec(Math.floor((Date.now() - runStartedAt) / 1000));
    tick();
    const id = window.setInterval(tick, 500);
    return () => window.clearInterval(id);
  }, [runningId, runStartedAt]);

  const filtered = useMemo(
    () =>
      filterGroup === "all"
        ? USER_PRESENTATION_QUERIES
        : USER_PRESENTATION_QUERIES.filter((q) => q.group === filterGroup),
    [filterGroup],
  );

  const counts = useMemo(() => {
    const c = { pass: 0, partial: 0, fail: 0, skip: 0, untested: 0 };
    for (const q of USER_PRESENTATION_QUERIES) {
      c[effectiveStatus(results[q.id])]++;
    }
    return c;
  }, [results]);

  const selected = selectedId ? USER_PRESENTATION_QUERIES.find((q) => q.id === selectedId) : null;
  const selectedResult = selectedId ? results[selectedId] : undefined;
  const busy = Boolean(runningId) || isGenerating || autoRunning;
  const running = Boolean(runningId);
  const phase = livePhase({ running, isGenerating, search: streamingSearch, assistantBuffer });

  const liveActivity = useMemo(() => {
    if (!running) return [];
    const slice = activityLog.filter((e) => e.ts >= activitySince);
    return (slice.length ? slice : activityLog).slice(0, 8);
  }, [activityLog, activitySince, running]);

  const composerIdle = modelReady && !isGenerating;

  const runOneQuery = useCallback(async (q: UserPresentationQuery): Promise<void> => {
    setError(null);
    if (!modelReady) {
      throw new Error("טען את המודל לפני בדיקה.");
    }
    setRunningId(q.id);
    setRunStartedAt(Date.now());
    setActivitySince(Date.now() - 500);
    setSelectedId(q.id);
    try {
      const turn = await qaChatBridge.ask(q.prompt, { newChat: true, forceLlm: forceLlmRef.current });
      const recorded: QaRecordedResult = {
        ...turn,
        autoStatus: autoGradeResult(turn, q),
        testedAt: Date.now(),
        runVersion: QA_CODE_VERSION,
      };
      setResults((prev) => ({ ...prev, [q.id]: recorded }));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setResults((prev) => ({
        ...prev,
        [q.id]: {
          query: q.prompt,
          reply: "",
          replySource: "unknown",
          usedModel: false,
          webContextSent: "",
          modelPromptOut: "",
          modelResponseIn: "",
          searchProviders: [],
          searchSummary: "",
          ms: 0,
          error: msg,
          autoStatus: "fail",
          testedAt: Date.now(),
          runVersion: QA_CODE_VERSION,
        },
      }));
      throw err;
    } finally {
      setRunningId(null);
      setRunStartedAt(null);
    }
  }, [modelReady]);

  const waitComposerIdle = useCallback(async (timeoutMs = 360_000) => {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      if (qaChatBridge.ready()) return;
      await new Promise((r) => window.setTimeout(r, 200));
    }
    throw new Error("timeout — המודל לא סיים בזמן");
  }, []);

  const runQuery = useCallback(
    async (q: UserPresentationQuery) => {
      if (busy && !autoRunning) return;
      try {
        await runOneQuery(q);
        await waitComposerIdle();
      } catch {
        /* recorded in runOneQuery */
      }
    },
    [autoRunning, busy, runOneQuery, waitComposerIdle],
  );

  const stopAutoRun = useCallback(() => {
    autoRunRef.current = false;
    setAutoRunning(false);
    setAutoProgress(null);
  }, []);

  const startAutoRun = useCallback(async (skipAlreadyTested = skipTested) => {
    if (!modelReady || autoRunning || runningId) return;
    const base =
      filterGroup === "all"
        ? USER_PRESENTATION_QUERIES
        : USER_PRESENTATION_QUERIES.filter((q) => q.group === filterGroup);
    const queue = skipAlreadyTested
      ? base.filter((q) => effectiveStatus(results[q.id]) === "untested")
      : base;

    if (!queue.length) {
      setError("אין שאלות לריצה (הכל כבר נבדק?).");
      return;
    }

    autoRunRef.current = true;
    setAutoRunning(true);
    setError(null);

    for (let i = 0; i < queue.length; i++) {
      if (!autoRunRef.current) break;
      const q = queue[i];
      setAutoProgress({ current: i + 1, total: queue.length, id: q.id });
      try {
        await runOneQuery(q);
        if (!autoRunRef.current) break;
        await waitComposerIdle();
        await new Promise((r) => window.setTimeout(r, 400));
      } catch {
        if (!autoRunRef.current) break;
        await waitComposerIdle().catch(() => {});
      }
    }

    stopAutoRun();
  }, [
    autoRunning,
    filterGroup,
    modelReady,
    results,
    runOneQuery,
    runningId,
    skipTested,
    stopAutoRun,
    waitComposerIdle,
  ]);

  const startAutoRunFromBeginning = useCallback(async () => {
    clearQaResults();
    setResults({});
    setSkipTested(false);
    setError(null);
    await startAutoRun(false);
  }, [startAutoRun]);

  useEffect(() => {
    if (!open) stopAutoRun();
  }, [open, stopAutoRun]);

  const setManualStatus = (id: string, status: Exclude<QaManualStatus, "untested">) => {
    setResults((prev) => {
      const existing = prev[id];
      if (!existing) {
        return {
          ...prev,
          [id]: {
            query: USER_PRESENTATION_QUERIES.find((q) => q.id === id)?.prompt ?? "",
            reply: "",
            replySource: "unknown",
            usedModel: false,
            webContextSent: "",
            modelPromptOut: "",
            modelResponseIn: "",
            searchProviders: [],
            searchSummary: "",
            ms: 0,
            autoStatus: "fail",
            manualStatus: status,
            testedAt: Date.now(),
          },
        };
      }
      return { ...prev, [id]: { ...existing, manualStatus: status } };
    });
  };

  const setNote = (id: string, note: string) => {
    setResults((prev) => {
      const existing = prev[id];
      if (!existing) return prev;
      return { ...prev, [id]: { ...existing, note } };
    });
  };

  const copyReport = async () => {
    const text = buildPresentationQaReport(USER_PRESENTATION_QUERIES, results);
    try {
      await navigator.clipboard.writeText(text);
      setCopyState("ok");
      window.setTimeout(() => setCopyState("idle"), 2000);
    } catch {
      setCopyState("fail");
      window.setTimeout(() => setCopyState("idle"), 2500);
    }
  };

  const downloadReport = () => {
    const text = buildPresentationQaReport(USER_PRESENTATION_QUERIES, results);
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `PRESENTATION_QA_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!open) return null;

  const copyLabel =
    copyState === "ok" ? "הועתק!" : copyState === "fail" ? "נכשל" : "העתק דוח";

  return (
    <div className="activity-overlay modal qa-panel-overlay" role="dialog" aria-modal="true">
      <div className="qa-panel modal-box">
        <header className="qa-panel-head">
          <div>
            <h2 id="qa-panel-title">בדיקת מצגת</h2>
            <p className="qa-panel-meta">
              {counts.pass}✅ {counts.partial}⚠️ {counts.fail}❌ · {counts.untested} נותרו
              {!modelReady ? " · מודל לא טעון" : ""}
            </p>
          </div>
          <div className="qa-panel-head-actions">
            <label className="qa-force-llm">
              <input type="checkbox" checked={forceLlm} onChange={(e) => setForceLlm(e.target.checked)} />
              force LLM
            </label>
            <button type="button" className="qa-btn" onClick={() => void copyReport()}>
              {copyLabel}
            </button>
            <button type="button" className="qa-btn" onClick={downloadReport}>
              MD
            </button>
            <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
              ×
            </button>
          </div>
        </header>

        <div className="qa-panel-toolbar">
          {(["all", "basic", "cross", "natural"] as const).map((g) => (
            <button
              key={g}
              type="button"
              className={`qa-filter-btn ${filterGroup === g ? "qa-filter-btn--active" : ""}`}
              onClick={() => setFilterGroup(g)}
            >
              {g === "all" ? "הכל" : GROUP_LABEL[g]}
            </button>
          ))}
          <button
            type="button"
            className="qa-btn qa-btn--ghost"
            onClick={() => {
              if (window.confirm("למחוק תוצאות?")) {
                clearQaResults();
                setResults({});
              }
            }}
          >
            נקה
          </button>
        </div>

        {error ? <p className="qa-panel-error">{error}</p> : null}

        <div className="qa-panel-body">
          <aside className="qa-auto-pane" dir="rtl" aria-label="ריצה אוטומטית">
            <h3 className="qa-auto-title">אוטומטי</h3>
            <p className="qa-auto-desc">
              שולח שאלה → מחכה שהמודל יסיים (כפתור שלח משתחרר) → הבאה.
            </p>
            {!autoRunning ? (
              <>
                <button
                  type="button"
                  className="qa-btn qa-btn--primary qa-auto-start"
                  disabled={!modelReady || busy}
                  onClick={() => void startAutoRunFromBeginning()}
                >
                  ↺ הרץ מההתחלה
                </button>
                <button
                  type="button"
                  className="qa-btn qa-btn--ghost qa-auto-start"
                  disabled={!modelReady || busy}
                  onClick={() => void startAutoRun()}
                >
                  ▶▶ הרץ רצף
                </button>
              </>
            ) : (
              <button type="button" className="qa-btn qa-btn--stop" onClick={stopAutoRun}>
                ⏹ עצור
              </button>
            )}
            <label className="qa-auto-check">
              <input
                type="checkbox"
                checked={skipTested}
                disabled={autoRunning}
                onChange={(e) => setSkipTested(e.target.checked)}
              />
              דלג על שנבדקו
            </label>
            {autoProgress ? (
              <div className="qa-auto-progress">
                <strong>
                  {autoProgress.current}/{autoProgress.total}
                </strong>
                <span>{autoProgress.id}</span>
              </div>
            ) : null}
            <div className={`qa-auto-idle ${composerIdle ? "qa-auto-idle--ok" : ""}`}>
              {composerIdle ? "● מוכן לשליחה" : "○ המודל עובד…"}
            </div>
          </aside>

          <nav className="qa-list-pane" dir="rtl" aria-label="רשימת שאלות">
            <ul className="qa-query-list">
              {filtered.map((q) => {
                const status = effectiveStatus(results[q.id]);
                const isRunning = runningId === q.id;
                return (
                  <li key={q.id} className={selectedId === q.id ? "qa-query-li--selected" : ""}>
                    <button
                      type="button"
                      className={`qa-query-row ${isRunning ? "qa-query-row--running" : ""}`}
                      onClick={() => setSelectedId(q.id)}
                    >
                      <span className="qa-query-status">{isRunning ? "⏳" : QA_STATUS_ICON[status]}</span>
                      <span className="qa-query-id">{q.id}</span>
                      <span className="qa-query-text">{q.prompt}</span>
                    </button>
                    <button
                      type="button"
                      className="qa-run-btn"
                      disabled={busy}
                      onClick={() => void runQuery(q)}
                      title="שלח"
                    >
                      ▶
                    </button>
                  </li>
                );
              })}
            </ul>
          </nav>

          {selected ? (
            <section className="qa-detail" dir="rtl">
              <div className="qa-detail-top">
                <div>
                  <span className="qa-query-id">{selected.id}</span>
                  <span className="qa-query-cat"> · {selected.category}</span>
                  <h3 className="qa-detail-prompt">{selected.prompt}</h3>
                </div>
                <button
                  type="button"
                  className="qa-btn qa-btn--primary"
                  disabled={busy}
                  onClick={() => void runQuery(selected)}
                >
                  {runningId === selected.id ? `${elapsedSec}ש'` : "▶ שלח"}
                </button>
              </div>

              {running && selectedId === runningId ? (
                <div className="qa-run-strip">
                  <span className="qa-run-dot" />
                  <strong>{phase.label || "רץ…"}</strong>
                  <span className="qa-run-elapsed">{elapsedSec}ש'</span>
                  <span className="qa-run-steps">
                    {STEPS.map((s, i) => (
                      <span
                        key={s}
                        className={`qa-run-step ${phase.step === i + 1 ? "qa-run-step--on" : ""} ${phase.step > i + 1 ? "qa-run-step--done" : ""}`}
                      >
                        {s}
                      </span>
                    ))}
                  </span>
                </div>
              ) : null}

              {running && (streamingSearch?.active || streamingSearch?.sources.length) ? (
                <SearchProgressPanel
                  active={!!streamingSearch.active}
                  query={streamingSearch.query}
                  sources={streamingSearch.sources}
                  summary={streamingSearch.summary}
                  brief={streamingSearch.brief}
                />
              ) : null}

              {running && assistantBuffer.trim() ? (
                <pre className="qa-stream-preview">{assistantBuffer}</pre>
              ) : null}

              {running && liveActivity.length ? (
                <ul className="qa-mini-log" dir="ltr">
                  {[...liveActivity].reverse().map((e) => (
                    <li key={e.id}>
                      <time>{formatActivityTime(e.ts)}</time> {directionLabel(e.direction)} {e.title}
                    </li>
                  ))}
                </ul>
              ) : null}

              <div className="qa-manual-row">
                {(["pass", "partial", "fail", "skip"] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    className={`qa-manual-btn ${effectiveStatus(selectedResult) === s ? "qa-manual-btn--active" : ""}`}
                    onClick={() => setManualStatus(selected.id, s)}
                  >
                    {QA_STATUS_ICON[s]}
                  </button>
                ))}
              </div>

              {selectedResult ? (
                <div className="qa-result-block">
                  <div className="qa-result-meta">
                    {QA_STATUS_ICON[selectedResult.autoStatus]} {selectedResult.ms}ms
                    {selectedResult.usedModel ? " · Gemma" : ""}
                    {selectedResult.searchProviders?.length
                      ? ` · ${selectedResult.searchProviders.join(", ")}`
                      : ""}
                  </div>
                  <textarea
                    className="qa-note-input"
                    rows={1}
                    value={selectedResult.note ?? ""}
                    onChange={(e) => setNote(selected.id, e.target.value)}
                    placeholder="הערה לדוח…"
                  />
                  {selectedResult.error ? <p className="qa-panel-error">{selectedResult.error}</p> : null}
                  <pre className="qa-detail-pre">{selectedResult.reply || "(ריק)"}</pre>
                  {selectedResult.webContextSent.trim() ? (
                    <details className="qa-web-ctx">
                      <summary>WEB CONTEXT</summary>
                      <pre className="qa-detail-pre qa-detail-pre--muted">
                        {selectedResult.webContextSent.slice(0, 4000)}
                      </pre>
                    </details>
                  ) : null}
                </div>
              ) : (
                <p className="qa-empty-hint">לחץ ▶ — התשובה תופיע כאן וגם בצ&apos;אט מאחורי הפאנל.</p>
              )}
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
