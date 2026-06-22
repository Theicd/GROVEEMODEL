import { useEffect, useState } from "react";
import {
  API_KEY_CATALOG,
  AISSTREAM_KEY_SAVED_EVENT,
  TAVILY_KEY_SAVED_EVENT,
  SCAVIO_KEY_SAVED_EVENT,
  getAisStreamApiKey,
  getScavioApiKey,
  getTavilyApiKey,
  listApiKeyEntries,
  setAisStreamApiKey,
  setScavioApiKey,
  setTavilyApiKey,
  isProviderEnabled,
  setProviderEnabled,
  getProviderUsage,
  resetProviderUsage,
  formatBytesKb,
  PROVIDER_ENABLED_EVENT,
  PROVIDER_USAGE_EVENT,
  type ApiKeyProviderId,
} from "./apiKeyStore";
import type { ProviderUsageRecord } from "./apiProviderUsage";
import { probeAisStreamConnection } from "../realityData/providers/aisStream";
import { probeScavioConnection } from "../webSearch/providers/scavio";
import { probeTavilyConnection } from "../webSearch/providers/tavily";
import "./apiKeys.css";

const KEY_PROVIDERS: ApiKeyProviderId[] = ["aisstream", "tavily", "scavio"];

const PROVIDER_ICON: Record<ApiKeyProviderId, string> = {
  aisstream: "⛴",
  tavily: "🔍",
  scavio: "🌐",
};

const PROVIDER_PLACEHOLDER: Record<ApiKeyProviderId, string> = {
  aisstream: "הדבק מפתח מ-aisstream.io",
  tavily: "הדבק מפתח מ-tavily.com",
  scavio: "הדבק מפתח sk_… מ-scavio.dev",
};

const SAVED_EVENT: Record<ApiKeyProviderId, string> = {
  aisstream: AISSTREAM_KEY_SAVED_EVENT,
  tavily: TAVILY_KEY_SAVED_EVENT,
  scavio: SCAVIO_KEY_SAVED_EVENT,
};

const readEnabledMap = (): Record<ApiKeyProviderId, boolean> => ({
  aisstream: isProviderEnabled("aisstream"),
  tavily: isProviderEnabled("tavily"),
  scavio: isProviderEnabled("scavio"),
});

const readUsageMap = (): Record<ApiKeyProviderId, ProviderUsageRecord> => ({
  aisstream: getProviderUsage("aisstream"),
  tavily: getProviderUsage("tavily"),
  scavio: getProviderUsage("scavio"),
});

const formatLastUsed = (ts?: number): string => {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("he-IL", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export function ApiKeysPanelContent({ active = true }: { active?: boolean }) {
  const [drafts, setDrafts] = useState<Record<ApiKeyProviderId, string>>(() => ({
    aisstream: getAisStreamApiKey() ?? "",
    tavily: getTavilyApiKey() ?? "",
    scavio: getScavioApiKey() ?? "",
  }));
  const [enabledMap, setEnabledMap] = useState(readEnabledMap);
  const [usageMap, setUsageMap] = useState(readUsageMap);
  const [savedFlash, setSavedFlash] = useState<ApiKeyProviderId | null>(null);
  const [testing, setTesting] = useState<ApiKeyProviderId | null>(null);
  const [testResult, setTestResult] = useState<string | null>(null);
  const entries = listApiKeyEntries();

  useEffect(() => {
    if (!active) return;
    const refresh = () => {
      setEnabledMap(readEnabledMap());
      setUsageMap(readUsageMap());
    };
    refresh();
    window.addEventListener(PROVIDER_ENABLED_EVENT, refresh);
    window.addEventListener(PROVIDER_USAGE_EVENT, refresh);
    return () => {
      window.removeEventListener(PROVIDER_ENABLED_EVENT, refresh);
      window.removeEventListener(PROVIDER_USAGE_EVENT, refresh);
    };
  }, [active]);

  if (!active) return null;

  const setDraft = (id: ApiKeyProviderId, value: string) =>
    setDrafts((prev) => ({ ...prev, [id]: value }));

  const saveKey = (id: ApiKeyProviderId) => {
    const value = drafts[id].trim();
    if (id === "aisstream") setAisStreamApiKey(value);
    else if (id === "tavily") setTavilyApiKey(value);
    else setScavioApiKey(value);
    window.dispatchEvent(new CustomEvent(SAVED_EVENT[id]));
    setSavedFlash(id);
    window.setTimeout(() => setSavedFlash(null), 2000);
  };

  const toggleEnabled = (id: ApiKeyProviderId, next: boolean) => {
    setProviderEnabled(id, next);
    setEnabledMap((prev) => ({ ...prev, [id]: next }));
  };

  const runTest = async (id: ApiKeyProviderId) => {
    const key = drafts[id].trim();
    if (!key) {
      setTestResult(`הזן מפתח ${API_KEY_CATALOG[id].labelHe} לפני בדיקה`);
      return;
    }
    setTesting(id);
    setTestResult(null);
    let message: string;
    if (id === "aisstream") {
      message = (await probeAisStreamConnection(key)).message;
    } else if (id === "tavily") {
      message = (await probeTavilyConnection(key)).message;
    } else {
      message = (await probeScavioConnection(key)).message;
    }
    setTesting(null);
    setTestResult(message);
    setUsageMap(readUsageMap());
  };

  const renderUsage = (id: ApiKeyProviderId) => {
    const usage = usageMap[id];
    const hasUsage = usage.requestCount > 0;
    const scavioCredits = id === "scavio" ? usage.creditsRemaining : undefined;

    return (
      <div className="api-keys-usage" aria-label="מד שימוש">
        <div className="api-keys-usage-head">
          <span className="api-keys-usage-title">שימוש</span>
          {hasUsage ? (
            <button
              type="button"
              className="api-keys-usage-reset"
              onClick={() => {
                resetProviderUsage(id);
                setUsageMap(readUsageMap());
              }}
            >
              אפס מונה
            </button>
          ) : null}
        </div>
        <div className="api-keys-usage-grid">
          <div className="api-keys-usage-stat">
            <span className="api-keys-usage-value">{usage.requestCount}</span>
            <span className="api-keys-usage-label">בקשות</span>
          </div>
          <div className="api-keys-usage-stat">
            <span className="api-keys-usage-value">{usage.totalHits}</span>
            <span className="api-keys-usage-label">תוצאות בסך הכל</span>
          </div>
          <div className="api-keys-usage-stat">
            <span className="api-keys-usage-value">{usage.lastHitCount ?? "—"}</span>
            <span className="api-keys-usage-label">תוצאות בבקשה אחרונה</span>
          </div>
          <div className="api-keys-usage-stat">
            <span className="api-keys-usage-value">
              {usage.lastBytesApprox ? formatBytesKb(usage.lastBytesApprox) : "—"}
            </span>
            <span className="api-keys-usage-label">נפח בבקשה אחרונה</span>
          </div>
          {scavioCredits != null ? (
            <div className="api-keys-usage-stat api-keys-usage-stat--wide">
              <span className="api-keys-usage-value">{scavioCredits}</span>
              <span className="api-keys-usage-label">קרדיטים נותרים (לפי Scavio)</span>
            </div>
          ) : null}
        </div>
        <p className="api-keys-usage-foot">
          {hasUsage
            ? `בקשה אחרונה: ${formatLastUsed(usage.lastRequestAt)} · ${usage.successCount} הצליחו`
            : "עדיין לא נרשמו בקשות מאז פתיחת המונה (כולל «בדוק חיבור»)."}
        </p>
      </div>
    );
  };

  const renderCard = (id: ApiKeyProviderId) => {
    const meta = API_KEY_CATALOG[id];
    const entry = entries.find((e) => e.id === id);
    const enabled = enabledMap[id];
    const hasKey = Boolean(entry?.configured);

    return (
      <section
        key={id}
        className={`api-keys-card${!enabled && hasKey ? " api-keys-card--off" : ""}`}
      >
        <div className="api-keys-card-head">
          <span className="api-keys-card-icon" aria-hidden="true">
            {PROVIDER_ICON[id]}
          </span>
          <div>
            <h3>{meta.labelHe}</h3>
            <p className="api-keys-card-hint">{meta.hintHe}</p>
          </div>
          {hasKey ? (
            enabled ? (
              <span className="api-keys-status api-keys-status--ok">פעיל · {entry?.masked}</span>
            ) : (
              <span className="api-keys-status api-keys-status--off">כבוי · {entry?.masked}</span>
            )
          ) : (
            <span className="api-keys-status">לא מוגדר</span>
          )}
        </div>

        <label className="api-keys-toggle" title="כיבוי מונע שימוש בקרדיטים בחיפוש וב-AIS">
          <input
            type="checkbox"
            checked={enabled}
            disabled={!hasKey}
            onChange={(e) => toggleEnabled(id, e.target.checked)}
          />
          <span className="api-keys-toggle-ui" aria-hidden="true" />
          <span className="api-keys-toggle-text">
            {enabled ? "מאגר פעיל — משתמש בקרדיטים" : "מאגר כבוי — לא יצרוך קרדיטים"}
          </span>
        </label>

        <label className="api-keys-field">
          <span>מפתח API</span>
          <input
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={drafts[id]}
            onChange={(e) => setDraft(id, e.target.value)}
            placeholder={PROVIDER_PLACEHOLDER[id]}
            dir="ltr"
          />
        </label>

        {renderUsage(id)}

        <div className="api-keys-actions">
          <button type="button" className="api-keys-btn api-keys-btn--primary" onClick={() => saveKey(id)}>
            {savedFlash === id ? "נשמר ✓" : "שמור מפתח"}
          </button>
          <button
            type="button"
            className="api-keys-btn"
            disabled={testing === id}
            onClick={() => void runTest(id)}
          >
            {testing === id ? "בודק…" : "בדוק חיבור"}
          </button>
          <a className="api-keys-link" href={meta.docsUrl} target="_blank" rel="noopener noreferrer">
            תיעוד →
          </a>
        </div>
        <p className="api-keys-note">המפתח נשמר מקומית · בקשות עוברות דרך proxy מקומי (npm run dev).</p>
      </section>
    );
  };

  return (
    <>
      <p className="api-keys-intro">
        <strong>AISStream</strong> לים · <strong>Tavily</strong> + <strong>Scavio</strong> לחיפוש אתרים ב-SERP
        (בנוסף ל-Wikipedia/GitHub/SearXNG). כבה מאגר שלא בשימוש כדי לחסוך קרדיטים.
      </p>

      {KEY_PROVIDERS.map(renderCard)}

      {testResult ? <p className="api-keys-test-result api-keys-test-result--global">{testResult}</p> : null}

      <section className="api-keys-card api-keys-card--soon">
        <h3>בקרוב</h3>
        <ul>
          <li>Cheapersal — מחירי סופר</li>
          <li>Pixabay — מדיה</li>
          <li>TMDB — סרטים</li>
        </ul>
      </section>
    </>
  );
}

type PanelProps = {
  open: boolean;
  onClose: () => void;
};

export function ApiKeysPanel({ open, onClose }: PanelProps) {
  if (!open) return null;

  return (
    <div
      className="settings-overlay modal api-keys-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="api-keys-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="settings-panel modal-box api-keys-panel">
        <div className="settings-head">
          <div className="settings-head-brand">
            <span className="settings-head-badge api-keys-head-badge" aria-hidden="true">
              🔑
            </span>
            <div>
              <h2 id="api-keys-title">מפתחות API</h2>
              <p className="settings-head-sub">AIS חי · חיפוש web (Tavily + Scavio Google)</p>
            </div>
          </div>
          <button type="button" className="icon-close settings-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </div>
        <ApiKeysPanelContent active />
      </div>
    </div>
  );
}
