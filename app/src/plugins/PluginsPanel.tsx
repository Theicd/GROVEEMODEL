import { useEffect, useState } from "react";
import { ApiKeysPanelContent } from "../apiKeys/ApiKeysPanel";
import "../apiKeys/apiKeys.css";
import { NewsEnginePanelContent } from "../groveeNews/NewsEnginePanel";
import "../groveeNews/newsEnginePanel.css";
import type { EngineStatus } from "../groveeNews/engine/types";
import { isNewsEngineBusy } from "../groveeNews/useNewsEngineStatus";
import {
  GROVEE_PLUGINS,
  getPluginHealthSnapshot,
  pollPluginHealth,
} from "./healthCoordinator";
import { PLUGIN_STATUS_EVENT } from "./events";
import type { GroveePlugin, PluginHealthState, PluginStatus } from "./types";
import type { PluginsHubTab } from "./pluginsHubTypes";
import "./plugins.css";

type Props = {
  open: boolean;
  onClose: () => void;
  tab?: PluginsHubTab;
  onTabChange?: (tab: PluginsHubTab) => void;
  healthSnapshot?: ReturnType<typeof getPluginHealthSnapshot>;
  newsEngineStatus?: EngineStatus;
  gemmaReady?: boolean;
  gemmaLoading?: boolean;
  gemmaLoadPct?: number;
  gemmaLoadDetail?: string;
  onRequestGemmaLoad?: () => void;
};

const HUB_TABS: { id: PluginsHubTab; label: string; icon: string; badgeClass?: string }[] = [
  { id: "plugins", label: "תוספים", icon: "🧩" },
  { id: "api-keys", label: "מפתחות API", icon: "🔑", badgeClass: "grovee-hub-tab-badge--keys" },
  { id: "rss", label: "RSS", icon: "📰", badgeClass: "grovee-hub-tab-badge--rss" },
];

const TAB_SUBTITLE: Record<PluginsHubTab, string> = {
  plugins: "שירותים מקומיים על המחשב — Grove Search Companion",
  "api-keys": "AIS חי · Tavily · Scavio — נשמר מקומית בדפדפן",
  rss: "מנוע חדשות ברקע — איסוף כותרות, אינדקס וסיכום Gemma",
};

const TAB_HEAD_BADGE: Record<PluginsHubTab, string> = {
  plugins: "🧩",
  "api-keys": "🔑",
  rss: "📰",
};

const statusLabel: Record<PluginStatus, string> = {
  unknown: "לא נבדק",
  offline: "לא פעיל",
  online: "פעיל",
  degraded: "פועל (מוגבל)",
};

const statusClass: Record<PluginStatus, string> = {
  unknown: "grovee-plugin-status--unknown",
  offline: "grovee-plugin-status--offline",
  online: "grovee-plugin-status--online",
  degraded: "grovee-plugin-status--degraded",
};

function PluginCard({
  plugin,
  health,
}: {
  plugin: GroveePlugin;
  health?: PluginHealthState;
}) {
  const [testing, setTesting] = useState<"health" | "search" | null>(null);
  const [testMessage, setTestMessage] = useState<string | null>(null);
  const [urlDraft, setUrlDraft] = useState(plugin.getBaseUrl());

  const status: PluginStatus = health?.status ?? "unknown";
  const download = plugin.download?.win;

  const runHealthCheck = async () => {
    setTesting("health");
    setTestMessage(null);
    await pollPluginHealth(plugin.id);
    window.dispatchEvent(new CustomEvent(PLUGIN_STATUS_EVENT));
    setTesting(null);
  };

  const runSearchTest = async () => {
    if (!plugin.probeSearch) return;
    setTesting("search");
    setTestMessage(null);
    const out = await plugin.probeSearch("webgpu browser ai");
    setTestMessage(out.messageHe);
    setTesting(null);
  };

  const saveUrl = () => {
    plugin.setBaseUrl(urlDraft.trim());
    void runHealthCheck();
  };

  const triggerDownload = (useFallback = false) => {
    const target = useFallback && download?.fallbackUrl ? download.fallbackUrl : download?.url;
    const name =
      useFallback && download?.fallbackFilename ? download.fallbackFilename : download?.filename;
    if (!target) return;
    const a = document.createElement("a");
    a.href = target;
    a.download = name ?? "download";
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  return (
    <section className="grovee-plugin-card">
      <div className="grovee-plugin-card-head">
        <span className="grovee-plugin-card-icon" aria-hidden="true">
          {plugin.icon}
        </span>
        <div className="grovee-plugin-card-titles">
          <h3>{plugin.nameHe}</h3>
          <p className="grovee-plugin-card-sub">{plugin.shortNameHe} · v{plugin.version}</p>
        </div>
        <span className={`grovee-plugin-status ${statusClass[status]}`}>
          ● {statusLabel[status]}
        </span>
      </div>

      <p className="grovee-plugin-desc">{plugin.descriptionHe}</p>

      {health?.messageHe ? (
        <p className="grovee-plugin-health-msg">{health.messageHe}</p>
      ) : null}

      {health?.enginesReady?.length ? (
        <p className="grovee-plugin-engines">
          מנועים: {health.enginesReady.join(", ")}
          {health.enginesFailed?.length ? ` · נכשלו: ${health.enginesFailed.join(", ")}` : ""}
        </p>
      ) : null}

      <ol className="grovee-plugin-steps">
        {plugin.installStepsHe.map((step, i) => (
          <li key={i}>{step}</li>
        ))}
      </ol>

      {download ? (
        <div className="grovee-plugin-download">
          <button
            type="button"
            className="grovee-plugin-btn grovee-plugin-btn--primary"
            onClick={() => triggerDownload(false)}
          >
            ⬇ הורדה ל-Windows (מתקין)
          </button>
          {download.fallbackUrl ? (
            <button
              type="button"
              className="grovee-plugin-btn"
              onClick={() => triggerDownload(true)}
            >
              חבילת ZIP
            </button>
          ) : null}
          <span className="grovee-plugin-download-hint">{download.sizeHintHe}</span>
        </div>
      ) : null}

      <label className="grovee-plugin-field">
        <span>כתובת שירות (ברירת מחדל {plugin.defaultBaseUrl})</span>
        <input
          type="url"
          value={urlDraft}
          onChange={(e) => setUrlDraft(e.target.value)}
          placeholder={plugin.defaultBaseUrl}
          dir="ltr"
          spellCheck={false}
        />
      </label>

      <div className="grovee-plugin-actions">
        <button type="button" className="grovee-plugin-btn" onClick={saveUrl}>
          שמור כתובת
        </button>
        <button
          type="button"
          className="grovee-plugin-btn"
          disabled={testing === "health"}
          onClick={() => void runHealthCheck()}
        >
          {testing === "health" ? "בודק…" : "בדוק חיבור"}
        </button>
        {plugin.probeSearch ? (
          <button
            type="button"
            className="grovee-plugin-btn grovee-plugin-btn--primary"
            disabled={testing === "search" || status === "offline"}
            onClick={() => void runSearchTest()}
          >
            {testing === "search" ? "מחפש…" : "בדוק חיפוש"}
          </button>
        ) : null}
      </div>

      {testMessage ? <p className="grovee-plugin-test-result">{testMessage}</p> : null}

      {plugin.upstream ? (
        <p className="grovee-plugin-upstream">
          מבוסס על{" "}
          <a href={plugin.upstream.url} target="_blank" rel="noopener noreferrer">
            {plugin.upstream.name} {plugin.upstream.version}
          </a>
        </p>
      ) : null}
    </section>
  );
}

function PluginsTabContent({
  snapshot,
}: {
  snapshot: ReturnType<typeof getPluginHealthSnapshot>;
}) {
  return (
    <>
      <p className="grovee-plugins-intro">
        תוספים מריצים תהליך קטן על המחשב שלך. GROVEEMODEL נשאר בדפדפן; התוסף מספק יכולות שלא
        אפשריות בדפדפן בלבד (למשל SERP scraping).
      </p>

      {GROVEE_PLUGINS.map((plugin) => (
        <PluginCard key={plugin.id} plugin={plugin} health={snapshot[plugin.id]} />
      ))}

      <section className="grovee-plugin-card grovee-plugin-card--soon">
        <h3>תוספים עתידיים</h3>
        <ul>
          <li>Cheapersal — מחירי סופר מקומי</li>
          <li>TMDB — קטalog סרטים</li>
          <li>HF Scanner — סריקת מודלים</li>
        </ul>
      </section>
    </>
  );
}

export function PluginsPanel({
  open,
  onClose,
  tab = "plugins",
  onTabChange,
  healthSnapshot,
  newsEngineStatus,
  gemmaReady,
  gemmaLoading,
  gemmaLoadPct,
  gemmaLoadDetail,
  onRequestGemmaLoad,
}: Props) {
  const snapshot = healthSnapshot ?? getPluginHealthSnapshot();
  const [activeTab, setActiveTab] = useState<PluginsHubTab>(tab);

  useEffect(() => {
    if (open) setActiveTab(tab);
  }, [open, tab]);

  const pickTab = (next: PluginsHubTab) => {
    setActiveTab(next);
    onTabChange?.(next);
  };

  if (!open) return null;

  const rssBusy = newsEngineStatus ? isNewsEngineBusy(newsEngineStatus) : false;
  const rssHeadlines =
    newsEngineStatus?.library?.rssHeadlines ?? newsEngineStatus?.rssHeadlines ?? 0;

  return (
    <div
      className="settings-overlay modal grovee-plugins-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="grovee-plugins-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="settings-panel modal-box grovee-plugins-panel grovee-plugins-panel--hub">
        <div className="settings-head">
          <div className="settings-head-brand">
            <span
              className={`settings-head-badge grovee-plugins-head-badge grovee-hub-head-badge grovee-hub-head-badge--${activeTab}`}
              aria-hidden="true"
            >
              {TAB_HEAD_BADGE[activeTab]}
            </span>
            <div>
              <h2 id="grovee-plugins-title">מרכז תוספים ושירותים</h2>
              <p className="settings-head-sub">{TAB_SUBTITLE[activeTab]}</p>
            </div>
          </div>
          <button type="button" className="icon-close settings-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </div>

        <nav className="grovee-hub-tabs" role="tablist" aria-label="לשוניות מרכז תוספים">
          {HUB_TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={activeTab === t.id}
              className={`grovee-hub-tab${activeTab === t.id ? " grovee-hub-tab--active" : ""}`}
              onClick={() => pickTab(t.id)}
            >
              <span className="grovee-hub-tab-icon" aria-hidden="true">
                {t.icon}
              </span>
              <span className="grovee-hub-tab-label">{t.label}</span>
              {t.id === "rss" && newsEngineStatus ? (
                rssBusy ? (
                  <span className={`grovee-hub-tab-badge ${t.badgeClass ?? ""} grovee-hub-tab-badge--live`}>
                    ●
                  </span>
                ) : rssHeadlines > 0 ? (
                  <span className={`grovee-hub-tab-badge ${t.badgeClass ?? ""}`}>
                    {rssHeadlines > 999 ? "999+" : rssHeadlines}
                  </span>
                ) : null
              ) : null}
            </button>
          ))}
        </nav>

        <div className="grovee-hub-body" role="tabpanel">
          {activeTab === "plugins" ? (
            <PluginsTabContent snapshot={snapshot} />
          ) : activeTab === "api-keys" ? (
            <ApiKeysPanelContent active />
          ) : (
            <NewsEnginePanelContent
              active
              gemmaReady={gemmaReady}
              gemmaLoading={gemmaLoading}
              gemmaLoadPct={gemmaLoadPct}
              gemmaLoadDetail={gemmaLoadDetail}
              onRequestGemmaLoad={onRequestGemmaLoad}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export type { PluginsHubTab } from "./pluginsHubTypes";
