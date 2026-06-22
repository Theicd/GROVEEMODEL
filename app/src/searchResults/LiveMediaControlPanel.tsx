import { useCallback, useEffect, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import {
  buildLiveMediaCatalogSummary,
  cancelLiveMediaValidation,
  syncAllLiveMediaSources,
  toggleLiveMediaSource,
  validateAllLiveMediaStreams,
} from "../liveMedia/catalogStore";
import type { LiveMediaCatalogSummary } from "../liveMedia/runtimeState";
import { subscribeLiveMediaSummary } from "../liveMedia/runtimeState";
import "./liveMediaControl.css";

type Props = {
  uiLang: ChatUiLanguage;
  open: boolean;
  onClose: () => void;
};

const emptySummary = (): LiveMediaCatalogSummary => ({
  channels: 0,
  radio: 0,
  channelStatus: { working: 0, warning: 0, offline: 0, unknown: 0 },
  radioStatus: { working: 0, warning: 0, offline: 0, unknown: 0 },
  categories: [],
  sources: [],
  lastSyncAt: null,
  progress: { phase: "idle", current: 0, total: 0, label: "" },
  lastError: null,
});

function formatTime(ts: number | null, uiLang: ChatUiLanguage): string {
  if (!ts) return uiLang === "he" ? "מעולם לא" : "Never";
  return new Date(ts).toLocaleString(uiLang === "he" ? "he-IL" : "en-US", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

export function LiveMediaControlPanel({ uiLang, open, onClose }: Props) {
  const [summary, setSummary] = useState<LiveMediaCatalogSummary>(emptySummary);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    void buildLiveMediaCatalogSummary().then(setSummary);
    return subscribeLiveMediaSummary(setSummary);
  }, [open]);

  const labels =
    uiLang === "he"
      ? {
          title: "בקרת TV LIVE / רדיו",
          sync: "סנכרון מקורות",
          validate: "בדיקת QA לסטרימים",
          cancel: "עצור QA",
          close: "סגור",
          radio: "תחנות רדיו",
          working: "עובדים",
          unknown: "לא נבדק",
          warning: "אזהרה",
          offline: "לא זמין",
          sources: "מקורות",
          categories: "קטגוריות TV",
          lastSync: "סנכרון אחרון",
          progressSync: "סורק מקורות",
          progressQa: "בודק סטרימים",
          enabled: "פעיל",
          count: "פריטים",
          error: "שגיאה",
        }
      : {
          title: "TV LIVE / Radio control",
          sync: "Sync sources",
          validate: "Run stream QA",
          cancel: "Stop QA",
          close: "Close",
          tv: "TV channels",
          radio: "Radio stations",
          working: "Working",
          unknown: "Unchecked",
          warning: "Warning",
          offline: "Offline",
          sources: "Sources",
          categories: "TV categories",
          lastSync: "Last sync",
          progressSync: "Syncing sources",
          progressQa: "Validating streams",
          enabled: "On",
          count: "items",
          error: "Error",
        };

  const runSync = useCallback(async () => {
    setBusy(true);
    try {
      await syncAllLiveMediaSources();
    } finally {
      setBusy(false);
      void buildLiveMediaCatalogSummary().then(setSummary);
    }
  }, []);

  const runValidate = useCallback(async () => {
    setBusy(true);
    try {
      await validateAllLiveMediaStreams();
    } finally {
      setBusy(false);
      void buildLiveMediaCatalogSummary().then(setSummary);
    }
  }, []);

  const onToggleSource = useCallback(async (id: string, enabled: boolean) => {
    await toggleLiveMediaSource(id, enabled);
    void buildLiveMediaCatalogSummary().then(setSummary);
  }, []);

  if (!open) return null;

  const { progress } = summary;
  const showProgress = progress.phase !== "idle" && progress.total > 0;
  const pct = showProgress ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <div className="serp-live-control-overlay" role="dialog" aria-modal="true" aria-label={labels.title}>
      <div className="serp-live-control-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="serp-live-control" dir={uiLang === "he" ? "rtl" : "ltr"}>
        <header className="serp-live-control-head">
          <h2>{labels.title}</h2>
          <button type="button" className="serp-live-control-close" onClick={onClose} aria-label={labels.close}>
            ×
          </button>
        </header>

        {summary.lastError ? (
          <p className="serp-live-control-error">
            {labels.error}: {summary.lastError}
          </p>
        ) : null}

        {showProgress ? (
          <div className="serp-live-control-progress">
            <div className="serp-live-control-progress-label">
              {progress.phase === "syncing" ? labels.progressSync : labels.progressQa}: {progress.label}
            </div>
            <div className="serp-live-control-progress-bar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
              <span style={{ width: `${pct}%` }} />
            </div>
            <div className="serp-live-control-progress-meta">
              {progress.current}/{progress.total} ({pct}%)
            </div>
          </div>
        ) : null}

        <div className="serp-live-control-actions">
          <button type="button" className="serp-live-control-btn" onClick={() => void runSync()} disabled={busy}>
            ↻ {labels.sync}
          </button>
          <button type="button" className="serp-live-control-btn serp-live-control-btn--qa" onClick={() => void runValidate()} disabled={busy}>
            ✓ {labels.validate}
          </button>
          {showProgress && progress.phase === "validating" ? (
            <button type="button" className="serp-live-control-btn" onClick={() => cancelLiveMediaValidation()}>
              ⏹ {labels.cancel}
            </button>
          ) : null}
        </div>

        <p className="serp-live-control-meta">
          {labels.lastSync}: {formatTime(summary.lastSyncAt, uiLang)}
        </p>

        <div className="serp-live-control-stats">
          <div className="serp-live-control-stat">
            <strong>{labels.tv}</strong>
            <span>{summary.channels}</span>
            <small>
              {labels.working} {summary.channelStatus.working} · {labels.unknown} {summary.channelStatus.unknown} ·{" "}
              {labels.offline} {summary.channelStatus.offline}
            </small>
          </div>
          <div className="serp-live-control-stat">
            <strong>{labels.radio}</strong>
            <span>{summary.radio}</span>
            <small>
              {labels.working} {summary.radioStatus.working} · {labels.unknown} {summary.radioStatus.unknown} ·{" "}
              {labels.offline} {summary.radioStatus.offline}
            </small>
          </div>
        </div>

        {summary.categories.length > 0 ? (
          <section className="serp-live-control-section">
            <h3>{labels.categories}</h3>
            <div className="serp-live-control-chips">
              {summary.categories.slice(0, 16).map((c) => (
                <span key={c.category} className="serp-live-control-chip">
                  {c.category} <em>{c.count}</em>
                </span>
              ))}
            </div>
          </section>
        ) : null}

        <section className="serp-live-control-section">
          <h3>{labels.sources}</h3>
          <ul className="serp-live-control-sources">
            {summary.sources.map((s) => (
              <li key={s.id}>
                <label className="serp-live-control-source-row">
                  <input
                    type="checkbox"
                    checked={s.enabled}
                    onChange={(e) => void onToggleSource(s.id, e.target.checked)}
                  />
                  <span className="serp-live-control-source-name">{s.name}</span>
                  <span className="serp-live-control-source-meta">
                    {s.type} · {s.channelCount || 0} {labels.count}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

export function LiveMediaStatusBadge({ uiLang }: { uiLang: ChatUiLanguage }) {
  const [summary, setSummary] = useState<LiveMediaCatalogSummary>(emptySummary);

  useEffect(() => {
    void buildLiveMediaCatalogSummary().then(setSummary);
    return subscribeLiveMediaSummary(setSummary);
  }, []);

  if (summary.progress.phase === "idle") {
    if (summary.channels + summary.radio === 0) {
      return (
        <span className="lm-live-status lm-live-status--empty">
          {uiLang === "he" ? "TV/רדיו: לא נטען" : "TV/Radio: not loaded"}
        </span>
      );
    }
    return (
      <span className="lm-live-status">
        {uiLang === "he" ? "TV" : "TV"} {summary.channels} · {uiLang === "he" ? "רדיו" : "Radio"} {summary.radio}
      </span>
    );
  }

  const pct = summary.progress.total
    ? Math.round((summary.progress.current / summary.progress.total) * 100)
    : 0;
  const verb =
    summary.progress.phase === "syncing"
      ? uiLang === "he"
        ? "סורק"
        : "Syncing"
      : uiLang === "he"
        ? "QA"
        : "QA";
  return (
    <span className="lm-live-status lm-live-status--busy">
      {verb} {pct}%
    </span>
  );
}
