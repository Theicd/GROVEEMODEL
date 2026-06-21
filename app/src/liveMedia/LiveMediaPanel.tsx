import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { channelToSearchHit, radioToSearchHit } from "./adapters";
import { LIVE_MEDIA_CATEGORIES, LIVE_MEDIA_COUNTRIES } from "./catalogs";
import {
  ensureLiveMediaLibrary,
  exportLiveMediaUserPrefs,
  hideChannelFromCatalog,
  hideRadioFromCatalog,
  importLiveMediaUserPrefs,
  restoreHiddenChannel,
  restoreHiddenRadio,
  syncAllLiveMediaSources,
  toggleChannelFavorite,
  toggleRadioFavorite,
} from "./catalogStore";
import type { Channel, RadioStation } from "./types";
import { collectLanguageCounts, languageDisplayLabel } from "./languageMetadata";
import type { LiveMediaUserPrefs } from "./userPrefs";
import { listRadioForPanel, listTvChannelsForPanel, searchLiveMediaChannels, searchLiveMediaRadio } from "./search";
import { subscribeLiveMediaSummary } from "./runtimeState";
import { LiveMediaResultsGrid } from "../searchResults/LiveMediaResultsGrid";
import { LiveMediaControlPanel, LiveMediaStatusBadge } from "../searchResults/LiveMediaControlPanel";
import type { UnifiedSearchHit } from "../searchResults/types";
import "./liveMediaPanel.css";

type HubView = "home" | "tv" | "radio" | "favorites" | "settings";

type Props = {
  uiLang: ChatUiLanguage;
  onClose: () => void;
};

const PAGE = 48;

export function LiveMediaPanel({ uiLang, onClose }: Props) {
  const [view, setView] = useState<HubView>("home");
  const [channels, setChannels] = useState<Channel[]>([]);
  const [radio, setRadio] = useState<RadioStation[]>([]);
  const [userPrefs, setUserPrefs] = useState<LiveMediaUserPrefs | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [country, setCountry] = useState("");
  const [language, setLanguage] = useState("");
  const [page, setPage] = useState(1);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const rtl = uiLang === "he";

  const L =
    uiLang === "he"
      ? {
          title: "TV LIVE / רדיו",
          home: "בית",
          tv: "טלוויזיה",
          radio: "רדיו",
          favorites: "מועדפים",
          settings: "הגדרות",
          close: "סגור",
          search: "חפש ערוצים / תחנות…",
          sync: "סנכרון",
          all: "הכל",
          country: "מדינה",
          language: "שפה",
          channels: "ערוצים",
          stations: "תחנות",
          categories: "קטגוריות",
          loadMore: "טען עוד",
          openSettings: "בקרה ו-QA",
          liveControl: "TV / רדיו",
          noFavorites: "אין מועדפים עדיין — סמן ☆ על ערוץ או תחנה.",
          hideChannel: "הסר לרשימה השחורה",
          prefsTitle: "מועדפים ורשימה שחורה",
          prefsHint:
            "מועדפים נשמרים גם ב-git (public/liveMedia/curatedFavorites.json) — ב-dev מתעדכנים אוטומטית כשמסמנים ☆. רשימה שחורה נשארת מקומית.",
          exportPrefs: "ייצוא JSON (גיבוי)",
          importPrefs: "ייבוא JSON",
          repoSyncHint: "ב-dev: commit את curatedFavorites.json אחרי עדכון מועדפים",
          blacklisted: "ברשימה השחורה",
          restore: "החזר",
          favSaved: "מועדפים",
          langFilter: "שפה",
          langUnknown: "לא מזוהה",
        }
      : {
          title: "TV LIVE / Radio",
          home: "Home",
          tv: "TV",
          radio: "Radio",
          favorites: "Favorites",
          settings: "Settings",
          close: "Close",
          search: "Search channels / stations…",
          sync: "Sync",
          all: "All",
          country: "Country",
          language: "Language",
          channels: "channels",
          stations: "stations",
          categories: "Categories",
          loadMore: "Load more",
          openSettings: "Control & QA",
          liveControl: "TV / Radio",
          noFavorites: "No favorites yet — tap ☆ on a channel or station.",
          hideChannel: "Add to blacklist",
          prefsTitle: "Favorites & blacklist",
          prefsHint:
            "Favorites sync to git (public/liveMedia/curatedFavorites.json) — auto-updated in dev when you star ☆. Blacklist stays local only.",
          exportPrefs: "Export JSON (backup)",
          importPrefs: "Import JSON",
          repoSyncHint: "In dev: commit curatedFavorites.json after updating favorites",
          blacklisted: "Blacklisted",
          restore: "Restore",
          favSaved: "Favorites",
          langFilter: "Language",
          langUnknown: "Unknown",
        };

  const refreshLibrary = useCallback(async () => {
    setLoading(true);
    try {
      const lib = await ensureLiveMediaLibrary();
      setChannels(lib.channels);
      setRadio(lib.radio);
      setUserPrefs(lib.prefs);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshLibrary();
    return subscribeLiveMediaSummary(() => {
      void refreshLibrary();
    });
  }, [refreshLibrary]);

  useEffect(() => {
    setPage(1);
  }, [view, query, category, country, language]);

  const tvListFull = useMemo(
    () =>
      listTvChannelsForPanel(channels, {
        category: category || undefined,
        country: country || undefined,
        language: language || undefined,
        query: query.trim() || undefined,
      }),
    [channels, category, country, language, query],
  );

  const tvHits: UnifiedSearchHit[] = useMemo(
    () => tvListFull.slice(0, page * PAGE).map((c) => channelToSearchHit(c)),
    [tvListFull, page],
  );

  const radioListFull = useMemo(
    () =>
      listRadioForPanel(radio, {
        country: country || undefined,
        language: language || undefined,
        query: query.trim() || undefined,
      }),
    [radio, country, language, query],
  );

  const radioHits: UnifiedSearchHit[] = useMemo(
    () => radioListFull.slice(0, page * PAGE).map((r) => radioToSearchHit(r)),
    [radioListFull, page],
  );

  const favoriteTvHits = useMemo(() => {
    let list = channels.filter((c) => c.favorite && (c.type === "tv" || c.type === "youtube"));
    if (query.trim()) list = searchLiveMediaChannels(list, query.trim(), 500);
    return list.map((c) => channelToSearchHit(c));
  }, [channels, query]);

  const favoriteRadioHits = useMemo(() => {
    let list = radio.filter((r) => r.favorite);
    if (query.trim()) list = searchLiveMediaRadio(list, query.trim(), 500);
    return list.map((r) => radioToSearchHit(r));
  }, [radio, query]);

  const favoriteIds = useMemo(() => {
    const ids = new Set<string>();
    for (const c of channels) {
      if (c.favorite) ids.add(`livetv-${c.id}`);
    }
    for (const r of radio) {
      if (r.favorite) ids.add(`radio-${r.id}`);
    }
    return ids;
  }, [channels, radio]);

  const favCount = favoriteTvHits.length + favoriteRadioHits.length;

  const handleToggleFavorite = useCallback(async (hit: UnifiedSearchHit) => {
    if (hit.kind === "livetv") {
      await toggleChannelFavorite(hit.id.replace(/^livetv-/, ""));
    } else if (hit.kind === "radio") {
      await toggleRadioFavorite(hit.id.replace(/^radio-/, ""));
    }
    await refreshLibrary();
  }, [refreshLibrary]);

  const handleHideChannel = useCallback(
    async (hit: UnifiedSearchHit) => {
      if (hit.kind === "livetv") {
        await hideChannelFromCatalog(hit.id.replace(/^livetv-/, ""));
      } else if (hit.kind === "radio") {
        await hideRadioFromCatalog(hit.id.replace(/^radio-/, ""));
      }
      await refreshLibrary();
    },
    [refreshLibrary],
  );

  const languageCounts = useMemo(() => collectLanguageCounts(channels, radio), [channels, radio]);

  const languageOptions = useMemo(() => {
    return [...languageCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 24)
      .map(([code, count]) => ({
        code,
        label: languageDisplayLabel(code, rtl),
        count,
      }));
  }, [languageCounts, rtl]);

  const categoryCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const c of channels) {
      if (c.type !== "tv" && c.type !== "youtube") continue;
      map.set(c.category || "general", (map.get(c.category || "general") ?? 0) + 1);
    }
    return map;
  }, [channels]);

  const countryCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const c of channels) {
      if (c.type !== "tv" && c.type !== "youtube") continue;
      const code = c.country || (c.source === "iptv-org-il" ? "il" : "");
      if (!code) continue;
      map.set(code, (map.get(code) ?? 0) + 1);
    }
    return map;
  }, [channels]);

  const runSync = useCallback(async () => {
    setLoading(true);
    try {
      await syncAllLiveMediaSources();
      await refreshLibrary();
    } finally {
      setLoading(false);
    }
  }, [refreshLibrary]);

  const nav: { id: HubView; label: string; badge?: number }[] = [
    { id: "home", label: L.home },
    { id: "tv", label: L.tv },
    { id: "radio", label: L.radio },
    { id: "favorites", label: L.favorites, badge: favCount || undefined },
    { id: "settings", label: L.settings },
  ];

  const gridFavProps = {
    favoriteIds,
    onToggleFavorite: handleToggleFavorite,
    onHideChannel: handleHideChannel,
  };

  return (
    <div className="lm-panel-inner" dir={rtl ? "rtl" : "ltr"}>
      <header className="lm-panel-head">
        <div className="lm-panel-title">
          <span className="lm-panel-dot" aria-hidden="true" />
          <span>{L.title}</span>
          <span className="lm-panel-meta">
            {channels.length} {L.tv} · {radio.length} {L.radio}
            <LiveMediaStatusBadge uiLang={uiLang} />
          </span>
        </div>
        <div className="lm-panel-head-actions">
          <button type="button" className="lm-panel-btn" onClick={() => void runSync()} disabled={loading}>
            ↻ {L.sync}
          </button>
          <button
            type="button"
            className="lm-panel-btn lm-panel-btn--primary"
            onClick={() => setSettingsOpen(true)}
            title={L.openSettings}
          >
            📺 {L.liveControl}
          </button>
          <button type="button" className="lm-panel-close" onClick={onClose} aria-label={L.close}>
            ×
          </button>
        </div>
      </header>

      <nav className="lm-nav" aria-label={L.title}>
        {nav.map((n) => (
          <button
            key={n.id}
            type="button"
            className={`lm-nav-btn${view === n.id ? " is-active" : ""}`}
            onClick={() => setView(n.id)}
          >
            {n.label}
            {n.badge ? <span className="lm-nav-badge">{n.badge}</span> : null}
          </button>
        ))}
      </nav>

      {(view === "tv" || view === "radio" || view === "home" || view === "favorites") && (
        <div className="lm-toolbar">
          <input
            type="search"
            className="lm-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={L.search}
            dir="auto"
          />
          {view !== "radio" && view !== "favorites" ? (
            <select
              className="lm-select"
              value={category}
              onChange={(e) => {
                setCategory(e.target.value);
                if (e.target.value) setView("tv");
              }}
            >
              <option value="">{L.all}</option>
              {LIVE_MEDIA_CATEGORIES.map((c) => (
                <option key={c.id} value={c.id}>
                  {rtl ? c.nameHe : c.name} ({categoryCounts.get(c.id) ?? 0})
                </option>
              ))}
            </select>
          ) : null}
          {view !== "favorites" ? (
            <select className="lm-select" value={country} onChange={(e) => setCountry(e.target.value)}>
              <option value="">{L.all}</option>
              {LIVE_MEDIA_COUNTRIES.map((c) => (
                <option key={c.code} value={c.code}>
                  {c.flag} {rtl ? c.nameHe : c.name} ({countryCounts.get(c.code) ?? 0})
                </option>
              ))}
            </select>
          ) : null}
          {view !== "favorites" ? (
            <select className="lm-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="">{L.all}</option>
              {languageOptions.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label} ({l.count})
                </option>
              ))}
            </select>
          ) : null}
        </div>
      )}

      <div className="lm-panel-body">
        {loading ? <div className="lm-loading">…</div> : null}

        {view === "home" && !loading ? (
          <div className="lm-home">
            <section>
              <h3>{L.categories}</h3>
              <div className="lm-chips">
                {LIVE_MEDIA_CATEGORIES.filter((c) => (categoryCounts.get(c.id) ?? 0) > 0).map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    className="lm-chip"
                    onClick={() => {
                      setCategory(c.id);
                      setView("tv");
                    }}
                  >
                    {rtl ? c.nameHe : c.name} <em>{categoryCounts.get(c.id)}</em>
                  </button>
                ))}
              </div>
            </section>
            {tvHits.length > 0 ? (
              <section>
                <h3>{L.tv}</h3>
                <LiveMediaResultsGrid hits={tvHits.slice(0, 12)} uiLang={uiLang} mode="livetv" {...gridFavProps} />
              </section>
            ) : null}
            {radioHits.length > 0 ? (
              <section>
                <h3>{L.radio}</h3>
                <LiveMediaResultsGrid hits={radioHits.slice(0, 12)} uiLang={uiLang} mode="radio" {...gridFavProps} />
              </section>
            ) : null}
          </div>
        ) : null}

        {view === "tv" && !loading ? (
          <>
            <p className="lm-count">
              {tvListFull.length} {L.channels}
            </p>
            <LiveMediaResultsGrid hits={tvHits} uiLang={uiLang} mode="livetv" {...gridFavProps} />
            {tvListFull.length > page * PAGE ? (
              <button type="button" className="lm-load-more" onClick={() => setPage((p) => p + 1)}>
                {L.loadMore}
              </button>
            ) : null}
          </>
        ) : null}

        {view === "radio" && !loading ? (
          <>
            <p className="lm-count">
              {radioListFull.length} {L.stations}
            </p>
            <LiveMediaResultsGrid hits={radioHits} uiLang={uiLang} mode="radio" {...gridFavProps} />
            {radioListFull.length > page * PAGE ? (
              <button type="button" className="lm-load-more" onClick={() => setPage((p) => p + 1)}>
                {L.loadMore}
              </button>
            ) : null}
          </>
        ) : null}

        {view === "favorites" && !loading ? (
          favCount === 0 ? (
            <p className="lm-empty-favorites">{L.noFavorites}</p>
          ) : (
            <div className="lm-favorites">
              {favoriteTvHits.length > 0 ? (
                <section>
                  <h3>
                    {L.tv} · {favoriteTvHits.length}
                  </h3>
                  <LiveMediaResultsGrid hits={favoriteTvHits} uiLang={uiLang} mode="livetv" {...gridFavProps} />
                </section>
              ) : null}
              {favoriteRadioHits.length > 0 ? (
                <section>
                  <h3>
                    {L.radio} · {favoriteRadioHits.length}
                  </h3>
                  <LiveMediaResultsGrid hits={favoriteRadioHits} uiLang={uiLang} mode="radio" {...gridFavProps} />
                </section>
              ) : null}
            </div>
          )
        ) : null}

        {view === "settings" && !loading ? (
          <div className="lm-settings">
            <section>
              <h3>{L.prefsTitle}</h3>
              <p>{L.prefsHint}</p>
              <p className="lm-settings-note">{L.repoSyncHint}</p>
              <div className="lm-settings-stat">
                ★ {L.favSaved}: {userPrefs?.favoriteChannelIds.length ?? 0} TV ·{" "}
                {userPrefs?.favoriteRadioIds.length ?? 0} {L.radio}
                <br />
                ✕ {L.blacklisted}: {userPrefs?.blacklistChannelIds.length ?? 0} TV ·{" "}
                {userPrefs?.blacklistRadioIds.length ?? 0} {L.radio}
              </div>
              <div className="lm-settings-actions">
                <button
                  type="button"
                  className="lm-panel-btn"
                  onClick={() => {
                    void exportLiveMediaUserPrefs().then((json) => {
                      void navigator.clipboard.writeText(json);
                    });
                  }}
                >
                  {L.exportPrefs}
                </button>
                <label className="lm-panel-btn">
                  {L.importPrefs}
                  <input
                    type="file"
                    accept="application/json,.json"
                    hidden
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      void file.text().then((raw) => importLiveMediaUserPrefs(raw)).then(() => refreshLibrary());
                      e.target.value = "";
                    }}
                  />
                </label>
                <button type="button" className="lm-panel-btn lm-panel-btn--primary" onClick={() => setSettingsOpen(true)}>
                  📺 {L.liveControl}
                </button>
              </div>
            </section>
            {(userPrefs?.blacklistChannelIds.length ?? 0) + (userPrefs?.blacklistRadioIds.length ?? 0) > 0 ? (
              <section>
                <h3>{L.blacklisted}</h3>
                <ul className="lm-blacklist-list">
                  {(userPrefs?.blacklistChannelIds ?? []).slice(0, 40).map((id) => (
                    <li key={`tv-${id}`}>
                      <span>TV · {id.slice(0, 12)}…</span>
                      <button type="button" onClick={() => void restoreHiddenChannel(id).then(() => refreshLibrary())}>
                        {L.restore}
                      </button>
                    </li>
                  ))}
                  {(userPrefs?.blacklistRadioIds ?? []).slice(0, 20).map((id) => (
                    <li key={`rd-${id}`}>
                      <span>{L.radio} · {id.slice(0, 12)}…</span>
                      <button type="button" onClick={() => void restoreHiddenRadio(id).then(() => refreshLibrary())}>
                        {L.restore}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </div>
        ) : null}
      </div>

      <LiveMediaControlPanel uiLang={uiLang} open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
