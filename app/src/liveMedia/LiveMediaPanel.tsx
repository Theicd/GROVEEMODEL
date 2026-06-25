import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { channelToSearchHit, radioToSearchHit } from "./adapters";
import {
  curatedSnapshotToChannel,
  fetchCuratedFavoritesFromRepo,
} from "./curatedFavorites";
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
import { languageDisplayLabel } from "./languageMetadata";
import { channelHasEnglish, channelHasHebrew, radioHasEnglish, radioHasHebrew } from "./heEnCatalogFilter";
import type { LiveMediaUserPrefs } from "./userPrefs";
import { listRadioForPanel, listTvChannelsForPanel, searchLiveMediaChannels, searchLiveMediaRadio } from "./search";
import { subscribeLiveMediaSummary } from "./runtimeState";
import { LiveMediaResultsGrid } from "../searchResults/LiveMediaResultsGrid";
import { LiveMediaControlPanel, LiveMediaStatusBadge } from "../searchResults/LiveMediaControlPanel";
import type { UnifiedSearchHit } from "../searchResults/types";
import { CableTunerView } from "./CableTunerView";
import { fetchStartupContext, getStartupContextSync } from "../startupContext";
import { buildRegionalRadioLineup } from "./cableTunerRadio";
import { channelQualityScore } from "./ranking";
import "./liveMediaPanel.css";

type HubView = "watch" | "browse" | "radio" | "settings";

type Props = {
  uiLang: ChatUiLanguage;
  onClose: () => void;
  /** Desktop: full workspace between sidebar and edge; mobile: side drawer */
  layout?: "side" | "full";
};

const PAGE = 48;

export function LiveMediaPanel({ uiLang, onClose, layout = "side" }: Props) {
  const isFullLayout = layout === "full";
  const [view, setView] = useState<HubView>("watch");
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
  const [curatedTvHits, setCuratedTvHits] = useState<UnifiedSearchHit[] | null>(null);
  const [curatedReady, setCuratedReady] = useState(false);
  const [geoCountry, setGeoCountry] = useState(() => getStartupContextSync()?.countryCode ?? "");
  const rtl = uiLang === "he";

  const L =
    uiLang === "he"
      ? {
          title: "TV LIVE / רדיו",
          watch: "שידור",
          browse: "חיפוש ערוצים",
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
          backChat: "חזרה לצ'אט",
          brandLive: "שידור חי",
          brandTag: "ערוצים · רדיו · מועדפים",
        }
      : {
          title: "TV LIVE / Radio",
          watch: "Watch",
          browse: "Browse channels",
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
          backChat: "Back to chat",
          brandLive: "LIVE",
          brandTag: "Channels · Radio · Favorites",
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
    let alive = true;
    void fetchStartupContext().then((ctx) => {
      if (!alive || !ctx?.countryCode) return;
      setGeoCountry(ctx.countryCode);
    });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (view !== "radio" || country) return;
    if (geoCountry) setCountry(geoCountry);
  }, [view, country, geoCountry]);

  useEffect(() => {
    let alive = true;
    void fetchCuratedFavoritesFromRepo().then((file) => {
      if (!alive) return;
      const hits = (file?.channels ?? []).map((snap) =>
        channelToSearchHit(curatedSnapshotToChannel(snap)),
      );
      setCuratedTvHits(hits);
      setCuratedReady(true);
    });
    return () => {
      alive = false;
    };
  }, []);

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
    const favIds = new Set(userPrefs?.favoriteChannelIds ?? []);
    const list = channels
      .filter(
        (c) =>
          (c.favorite || favIds.has(c.id)) &&
          (c.type === "tv" || c.type === "youtube"),
      )
      .sort((a, b) => channelQualityScore(b) - channelQualityScore(a));
    return list.map((c) => channelToSearchHit(c));
  }, [channels, userPrefs?.favoriteChannelIds]);

  const tunerFavorites = useMemo(() => {
    if (curatedTvHits && curatedTvHits.length > 0) return curatedTvHits;
    return favoriteTvHits;
  }, [curatedTvHits, favoriteTvHits]);

  const regionalRadio = useMemo(
    () => buildRegionalRadioLineup(radio, geoCountry || "us"),
    [radio, geoCountry],
  );

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

  const languageOptions = useMemo(() => {
    const hebTv = channels.filter(
      (c) => (c.type === "tv" || c.type === "youtube") && channelHasHebrew(c),
    ).length;
    const engTv = channels.filter(
      (c) => (c.type === "tv" || c.type === "youtube") && channelHasEnglish(c),
    ).length;
    const hebRadio = radio.filter((r) => radioHasHebrew(r)).length;
    const engRadio = radio.filter((r) => radioHasEnglish(r)).length;
    const opts = [
      { code: "heb", label: languageDisplayLabel("heb", rtl), count: hebTv + hebRadio },
      { code: "eng", label: languageDisplayLabel("eng", rtl), count: engTv + engRadio },
    ];
    return opts.filter((o) => o.count > 0);
  }, [channels, radio, rtl]);

  const browsableCategories = useMemo(
    () => LIVE_MEDIA_CATEGORIES.filter((c) => c.id !== "news"),
    [],
  );

  const categoryCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const c of channels) {
      if (c.type !== "tv" && c.type !== "youtube") continue;
      const cat = c.category || "general";
      if (cat === "news") continue;
      map.set(cat, (map.get(cat) ?? 0) + 1);
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
    { id: "watch", label: L.watch, badge: tunerFavorites.length || undefined },
    { id: "browse", label: L.browse },
    { id: "radio", label: L.radio },
    { id: "settings", label: L.settings },
  ];

  const gridFavProps = {
    favoriteIds,
    onToggleFavorite: handleToggleFavorite,
    onHideChannel: handleHideChannel,
  };

  return (
    <div
      className={`lm-panel-inner${isFullLayout && view !== "watch" ? " lm-panel-inner--full" : ""}${view === "watch" ? " lm-panel-inner--tuner-only" : ""}`}
      dir={rtl ? "rtl" : "ltr"}
    >
      {view !== "watch" ? (
      <header className={isFullLayout ? "lm-hero" : "lm-panel-head"}>
        {isFullLayout ? (
          <>
            <div className="lm-hero-brand">
              <span className="lm-hero-eyebrow">GROVEE</span>
              <span className="lm-hero-title">{L.brandLive}</span>
              <span className="lm-hero-sub">{L.brandTag}</span>
            </div>
            <div className="lm-hero-stats">
              <span>
                <strong>{channels.length}</strong> {L.tv}
              </span>
              <span>
                <strong>{radio.length}</strong> {L.radio}
              </span>
              <LiveMediaStatusBadge uiLang={uiLang} />
            </div>
            <div className="lm-hero-actions">
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
              <button type="button" className="lm-back-chat" onClick={onClose}>
                ← {L.backChat}
              </button>
            </div>
          </>
        ) : (
          <>
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
          </>
        )}
      </header>
      ) : null}

      {view !== "watch" ? (
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
      ) : null}

      {view === "watch" ? (
        <div className="lm-panel-body lm-panel-body--watch">
          <CableTunerView
            favorites={tunerFavorites}
            regionalRadio={regionalRadio}
            uiLang={uiLang}
            loading={!curatedReady && loading}
            onOpenBrowse={() => setView("browse")}
            onRemoveFavorite={handleToggleFavorite}
            onBack={isFullLayout ? undefined : onClose}
          />
        </div>
      ) : (
        <>
          {(view === "browse" || view === "radio") && (
            <div className="lm-toolbar">
              <input
                type="search"
                className="lm-search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={L.search}
                dir="auto"
              />
              {view === "browse" ? (
                <select className="lm-select" value={category} onChange={(e) => setCategory(e.target.value)}>
                  <option value="">{L.all}</option>
                  {browsableCategories.map((c) => (
                    <option key={c.id} value={c.id}>
                      {rtl ? c.nameHe : c.name} ({categoryCounts.get(c.id) ?? 0})
                    </option>
                  ))}
                </select>
              ) : null}
              <select className="lm-select" value={country} onChange={(e) => setCountry(e.target.value)}>
                <option value="">{L.all}</option>
                {LIVE_MEDIA_COUNTRIES.filter((c) => (countryCounts.get(c.code) ?? 0) > 0).map((c) => (
                  <option key={c.code} value={c.code}>
                    {c.flag} {rtl ? c.nameHe : c.name} ({countryCounts.get(c.code) ?? 0})
                  </option>
                ))}
              </select>
              <select className="lm-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option value="">{L.all}</option>
                {languageOptions.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label} ({l.count})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="lm-content-shell">
            <div className="lm-panel-body">
              {loading ? <div className="lm-loading">…</div> : null}

              {view === "browse" && !loading ? (
                <>
                  <p className="lm-count">
                    {tvListFull.length} {L.channels} · ★ {tunerFavorites.length} {L.favorites}
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
          </div>
        </>
      )}

      <LiveMediaControlPanel uiLang={uiLang} open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
