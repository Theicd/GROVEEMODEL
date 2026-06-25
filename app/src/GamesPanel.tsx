import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GameCard } from "./GameCard";
import { GamesHeroCarousel } from "./GamesHeroCarousel";
import { GAME_CATEGORIES } from "./gameSearch/archiveQueries";
import { buildHeroLineup } from "./gameSearch/gameThumbnail";
import {
  GAMES_CATALOG_PAGE_SIZE,
  loadMoreOnlineGames,
  parseGameUserRequest,
  randomOnlineGames,
  searchOnlineGamesWithFallback,
  type GameCategoryId,
  type OnlineGame,
} from "./gameSearch";
import {
  filterBlacklistedGames,
  getBlacklistedIds,
  getFavoriteGames,
  getFavoriteIds,
  getRecentPlayedGames,
  loadGamesSession,
  toggleBlacklistedGame,
  toggleFavoriteGame,
} from "./localExperience/gamesStore";
import { useNetworkStatus } from "./hooks/useNetworkStatus";

type PanelView = "browse" | "recent" | "favorites";

export type GamesPanelLayout = "side" | "full";

const ARCHIVE_EMBED_WIDTH = 560;
const ARCHIVE_EMBED_HEIGHT = 384;
const GAME_EMBED_LAYOUT_DELAY_MS = 360;

type Props = {
  games: OnlineGame[];
  loading: boolean;
  embedGame: OnlineGame | null;
  title?: string;
  initialCategory?: GameCategoryId | null;
  startView?: PanelView;
  /** side = third column beside chat; full = workspace until sidebar (desktop). */
  layout?: GamesPanelLayout;
  onExpandFull?: () => void;
  onShrinkSide?: () => void;
  onClose: () => void;
  onPlay: (game: OnlineGame) => void;
  onBackFromEmbed: () => void;
  onGamesUpdate: (games: OnlineGame[], title?: string, category?: GameCategoryId | null) => void;
  onLoadingChange: (loading: boolean) => void;
};

export function GamesPanel({
  games,
  loading,
  embedGame,
  title = "משחקים און־ליין",
  initialCategory = null,
  startView = "browse",
  layout = "side",
  onExpandFull,
  onShrinkSide,
  onClose,
  onPlay,
  onBackFromEmbed,
  onGamesUpdate,
  onLoadingChange,
}: Props) {
  const network = useNetworkStatus();
  const offline = network === "offline" || network === "limited";
  const [searchInput, setSearchInput] = useState("");
  const [expandedSearch, setExpandedSearch] = useState(false);
  const [activeCategory, setActiveCategory] = useState<GameCategoryId>(initialCategory ?? "featured");
  const [panelView, setPanelView] = useState<PanelView>("browse");
  const [storedGames, setStoredGames] = useState<OnlineGame[]>([]);
  const [favoriteIds, setFavoriteIds] = useState<Set<string>>(new Set());
  const [blacklistIds, setBlacklistIds] = useState<Set<string>>(new Set());
  const [heroFavorites, setHeroFavorites] = useState<OnlineGame[]>([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [canLoadMore, setCanLoadMore] = useState(true);
  const embedStageRef = useRef<HTMLDivElement | null>(null);
  const [embedSrc, setEmbedSrc] = useState<string | null>(null);
  const [embedScale, setEmbedScale] = useState(1);

  const refreshFavorites = useCallback(async () => {
    setFavoriteIds(await getFavoriteIds());
    setHeroFavorites(await getFavoriteGames());
  }, []);

  const refreshBlacklist = useCallback(async () => {
    setBlacklistIds(await getBlacklistedIds());
  }, []);

  useEffect(() => {
    void refreshFavorites();
    void refreshBlacklist();
  }, [refreshBlacklist, refreshFavorites]);

  useEffect(() => {
    if (initialCategory) setActiveCategory(initialCategory);
  }, [initialCategory]);

  useEffect(() => {
    if (!embedGame) {
      setEmbedSrc(null);
      return;
    }

    setEmbedSrc(null);
    const id = window.setTimeout(() => {
      setEmbedSrc(embedGame.embedUrl);
    }, layout === "full" ? GAME_EMBED_LAYOUT_DELAY_MS : 0);
    return () => window.clearTimeout(id);
  }, [embedGame, layout]);

  useEffect(() => {
    if (!embedGame) return;
    const stage = embedStageRef.current;
    if (!stage) return;

    const updateScale = () => {
      const rect = stage.getBoundingClientRect();
      const next = Math.min(
        rect.width / ARCHIVE_EMBED_WIDTH,
        rect.height / ARCHIVE_EMBED_HEIGHT,
      );
      setEmbedScale(Math.max(0.5, next || 1));
    };

    updateScale();
    const observer = new ResizeObserver(updateScale);
    observer.observe(stage);
    window.addEventListener("resize", updateScale);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateScale);
    };
  }, [embedGame]);

  const loadStoredView = useCallback(async (view: PanelView) => {
    setPanelView(view);
    if (view === "recent") {
      setStoredGames(await getRecentPlayedGames());
    } else if (view === "favorites") {
      setStoredGames(await getFavoriteGames());
    }
  }, []);

  useEffect(() => {
    if (startView === "browse") {
      setPanelView("browse");
      return;
    }
    void loadStoredView(startView);
  }, [loadStoredView, startView]);

  const filterStoredGames = useCallback((list: OnlineGame[]) => {
    const q = searchInput.trim().toLowerCase();
    if (!q) return list;
    return list.filter((g) => g.title.toLowerCase().includes(q) || g.description?.toLowerCase().includes(q));
  }, [searchInput]);

  const runSearch = useCallback(
    async (query: string, category: GameCategoryId | null = null) => {
      if (offline) {
        const session = await loadGamesSession();
        if (session?.games.length) {
          onGamesUpdate(session.games, `${session.title} (מקומי)`, session.category);
        }
        return;
      }
      setPanelView("browse");
      onLoadingChange(true);
      try {
        const resolved = parseGameUserRequest(query);
        if (category) resolved.category = category;
        const result = await searchOnlineGamesWithFallback(resolved, 16);
        onGamesUpdate(result.games, query ? `חיפוש: ${query}` : resolved.panelTitle, category);
      } finally {
        onLoadingChange(false);
      }
    },
    [offline, onGamesUpdate, onLoadingChange],
  );

  const loadCategory = useCallback(
    async (cat: GameCategoryId) => {
      setPanelView("browse");
      setActiveCategory(cat);
      if (offline) {
        const session = await loadGamesSession();
        if (session?.games.length) {
          onGamesUpdate(session.games, `${session.title} (מקומי)`, session.category);
        }
        return;
      }
      onLoadingChange(true);
      setCanLoadMore(true);
      try {
        const result = await randomOnlineGames(GAMES_CATALOG_PAGE_SIZE, cat);
        onGamesUpdate(result.games, GAME_CATEGORIES.find((c) => c.id === cat)?.labelHe, cat);
      } finally {
        onLoadingChange(false);
      }
    },
    [offline, onGamesUpdate, onLoadingChange],
  );

  const refreshList = useCallback(async () => {
    if (panelView === "recent") {
      await loadStoredView("recent");
      return;
    }
    if (panelView === "favorites") {
      await loadStoredView("favorites");
      return;
    }
    if (offline) {
      const session = await loadGamesSession();
      if (session?.games.length) {
        onGamesUpdate(session.games, `${session.title} (מקומי)`, session.category);
      }
      return;
    }
    onLoadingChange(true);
    setCanLoadMore(true);
    try {
      const result = await randomOnlineGames(GAMES_CATALOG_PAGE_SIZE, activeCategory);
      onGamesUpdate(result.games, GAME_CATEGORIES.find((c) => c.id === activeCategory)?.labelHe, activeCategory);
    } finally {
      onLoadingChange(false);
    }
  }, [activeCategory, loadStoredView, offline, onGamesUpdate, onLoadingChange, panelView]);

  const loadMoreBrowse = useCallback(async () => {
    if (offline || panelView !== "browse" || loadingMore || !canLoadMore) return;
    setLoadingMore(true);
    try {
      const result = await loadMoreOnlineGames(GAMES_CATALOG_PAGE_SIZE, activeCategory, games.map((g) => g.id));
      if (!result.games.length) {
        setCanLoadMore(false);
        return;
      }
      const merged = [...games];
      const seen = new Set(games.map((g) => g.id.toLowerCase()));
      for (const g of result.games) {
        const key = g.id.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(g);
      }
      onGamesUpdate(
        merged,
        GAME_CATEGORIES.find((c) => c.id === activeCategory)?.labelHe,
        activeCategory,
      );
      if (result.games.length < GAMES_CATALOG_PAGE_SIZE) setCanLoadMore(false);
    } finally {
      setLoadingMore(false);
    }
  }, [activeCategory, canLoadMore, games, loadingMore, offline, onGamesUpdate, panelView]);

  const handleToggleFavorite = useCallback(
    async (game: OnlineGame) => {
      await toggleFavoriteGame(game);
      await refreshFavorites();
      if (panelView === "favorites") {
        setStoredGames(await getFavoriteGames());
      }
    },
    [panelView, refreshFavorites],
  );

  const handleToggleBlacklist = useCallback(
    async (game: OnlineGame) => {
      const nowBlacklisted = await toggleBlacklistedGame(game);
      await refreshBlacklist();
      if (nowBlacklisted && panelView === "browse" && !title.startsWith("חיפוש:")) {
        onGamesUpdate(
          filterBlacklistedGames(games, new Set([...blacklistIds, game.id])),
          title,
          activeCategory,
        );
      }
      if (nowBlacklisted) {
        setHeroFavorites((prev) => prev.filter((g) => g.id !== game.id));
      }
    },
    [activeCategory, blacklistIds, games, onGamesUpdate, panelView, refreshBlacklist, title],
  );

  const requestEmbedFullscreen = useCallback(async () => {
    const stage = embedStageRef.current;
    if (!stage) return;
    try {
      if (!document.fullscreenElement) {
        await stage.requestFullscreen();
      }
    } catch {
      /* Browser may reject fullscreen outside a direct user gesture. */
    }
  }, []);

  useEffect(() => {
    if (!embedGame) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (document.fullscreenElement !== embedStageRef.current) return;

      if (event.key === "Backspace") {
        event.preventDefault();
        event.stopPropagation();
        void document.exitFullscreen();
        return;
      }

      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
      }
    };

    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [embedGame]);

  const browseGames =
    panelView === "browse" && !title.startsWith("חיפוש:")
      ? filterBlacklistedGames(games, blacklistIds)
      : games;
  const displayGames = panelView === "browse" ? browseGames : filterStoredGames(storedGames);
  const displayLoading = panelView === "browse" && loading;
  const favCount = favoriteIds.size;
  const heroLineup = useMemo(
    () =>
      panelView === "browse"
        ? buildHeroLineup(filterBlacklistedGames(heroFavorites, blacklistIds))
        : [],
    [blacklistIds, heroFavorites, panelView],
  );
  const activeCategoryMeta = GAME_CATEGORIES.find((c) => c.id === activeCategory);
  const catalogSectionTitle = activeCategoryMeta ? `${activeCategoryMeta.icon} ${activeCategoryMeta.labelHe}` : title;
  const showHero = !embedGame && panelView === "browse" && heroLineup.length > 0 && !displayLoading;
  const isCategoryBrowse = panelView === "browse" && !title.startsWith("חיפוש:");

  return (
    <div className={`games-panel-inner${layout === "full" ? " games-panel-inner--full" : ""}`}>
      <header className="games-panel-head">
        <div className="games-panel-title">
          <span className="games-panel-dot" aria-hidden="true" />
          {embedGame ? embedGame.title : panelView === "recent" ? "שיחקת לאחרונה" : panelView === "favorites" ? "מועדפים" : title}
        </div>
        <div className="games-panel-head-actions">
          {embedGame ? (
            <>
              <button type="button" className="games-panel-btn" onClick={onBackFromEmbed}>
                ← רשימה
              </button>
              <button
                type="button"
                className="games-panel-btn games-panel-btn--expand"
                onClick={(e) => {
                  e.currentTarget.blur();
                  void requestEmbedFullscreen();
                }}
                title="ALT+ENTER בתוך המשחק"
              >
                ⛶ מסך מלא
              </button>
            </>
          ) : (
            <>
              <button type="button" className="games-panel-btn" onClick={() => void refreshList()}>
                🔄 רענן
              </button>
              <button
                type="button"
                className="games-panel-btn"
                onClick={() => setExpandedSearch((v) => !v)}
              >
                {expandedSearch ? "סגור חיפוש" : "🔍 חיפוש"}
              </button>
              {layout === "side" && onExpandFull ? (
                <button
                  type="button"
                  className="games-panel-btn games-panel-btn--expand"
                  onClick={onExpandFull}
                  title="פתיחה מלאה עד התפריט"
                  aria-label="פתיחה מלאה"
                >
                  ⛶ מלא
                </button>
              ) : null}
              {layout === "full" && onShrinkSide ? (
                <button
                  type="button"
                  className="games-panel-btn games-panel-btn--expand"
                  onClick={onShrinkSide}
                  title="חזרה לפאנל לצד הצ'אט"
                  aria-label="צמצם לפאנל צד"
                >
                  ⊟ צד
                </button>
              ) : null}
            </>
          )}
          <button type="button" className="games-panel-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </div>
      </header>

      {offline && !embedGame ? (
        <p className="games-panel-offline-hint">📵 מצב מקומי — מציג משחקים שמורים. ניגון דורש Archive.org.</p>
      ) : null}

      {embedGame ? (
        <div className="games-panel-embed-wrap">
          <div ref={embedStageRef} className="games-panel-embed-stage">
            {embedSrc ? (
              <div
                className="games-panel-embed-frame"
                style={{
                  width: `${ARCHIVE_EMBED_WIDTH * embedScale}px`,
                  height: `${ARCHIVE_EMBED_HEIGHT * embedScale}px`,
                }}
              >
                <iframe
                  key={`${embedGame.id}-${layout}-${embedSrc}`}
                  className="games-panel-embed"
                  src={embedSrc}
                  title={embedGame.title}
                  allow="fullscreen; autoplay; gamepad"
                  allowFullScreen
                  sandbox="allow-scripts allow-same-origin allow-popups allow-forms allow-pointer-lock"
                  style={{
                    width: ARCHIVE_EMBED_WIDTH,
                    height: ARCHIVE_EMBED_HEIGHT,
                    transform: `scale(${embedScale})`,
                  }}
                />
              </div>
            ) : (
              <p className="games-panel-embed-loading">מכין את המשחק לגודל מלא…</p>
            )}
          </div>
          <a
            className="games-panel-external"
            href={embedGame.playUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            פתח ב-Archive.org ↗
          </a>
        </div>
      ) : (
        <div className="games-panel-scroll">
          {expandedSearch || panelView !== "browse" ? (
            <div className="games-panel-search-bar">
              <input
                type="search"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder={
                  panelView === "favorites"
                    ? "חפש במועדפים…"
                    : panelView === "recent"
                      ? "חפש באחרונים…"
                      : "חפש משחק… (ארקייד, pacman, shooter)"
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter" && panelView === "browse") void runSearch(searchInput, activeCategory);
                }}
              />
              {panelView === "browse" ? (
                <button type="button" onClick={() => void runSearch(searchInput, activeCategory)}>
                  חפש
                </button>
              ) : null}
            </div>
          ) : null}

          <div className="games-panel-categories games-panel-categories--library" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={panelView === "recent"}
              className={`games-panel-cat games-panel-cat--library${panelView === "recent" ? " active" : ""}`}
              onClick={() => void loadStoredView("recent")}
            >
              🕐 אחרונים
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={panelView === "favorites"}
              className={`games-panel-cat games-panel-cat--library${panelView === "favorites" ? " active" : ""}`}
              onClick={() => void loadStoredView("favorites")}
            >
              ★ מועדפים{favCount > 0 ? ` (${favCount})` : ""}
            </button>
          </div>

          {showHero ? (
            <GamesHeroCarousel games={heroLineup} favoriteCount={favCount} layout={layout} onPlay={onPlay} />
          ) : null}

          {panelView === "browse" ? (
            <section className="games-browse-toolbar" aria-label="עיון לפי קטגוריה">
              <div className="games-browse-toolbar-head">
                <h3 className="games-browse-toolbar-title">עיון לפי קטגוריה</h3>
                <p className="games-browse-toolbar-sub">בחר ז&apos;אנר — הרשימה למטה מתעדכנת</p>
              </div>
              <label className="games-category-select-wrap">
                <span className="games-category-select-label">קטגוריה</span>
                <select
                  className="games-category-select"
                  value={activeCategory}
                  onChange={(e) => void loadCategory(e.target.value as GameCategoryId)}
                  aria-label="בחירת קטגוריית משחקים"
                >
                  {GAME_CATEGORIES.map((cat) => (
                    <option key={cat.id} value={cat.id}>
                      {cat.icon} {cat.labelHe}
                    </option>
                  ))}
                </select>
              </label>
            </section>
          ) : null}

          <div className="games-panel-body">
            {panelView === "browse" && !displayLoading && displayGames.length > 0 ? (
              <header className="games-catalog-section-head">
                <h3 className="games-catalog-section-title">{catalogSectionTitle}</h3>
                <span className="games-catalog-section-count">{displayGames.length} משחקים</span>
              </header>
            ) : null}
            {displayLoading ? (
              <p className="games-panel-status">טוען משחקים…</p>
            ) : displayGames.length === 0 ? (
              <p className="games-panel-status">
                {panelView === "recent"
                  ? searchInput.trim()
                    ? "אין התאמות באחרונים."
                    : "עדיין לא שיחקת — לחץ ▶ על משחק כדי לשמור כאן."
                  : panelView === "favorites"
                    ? searchInput.trim()
                      ? "אין התאמות במועדפים."
                      : "אין מועדפים — לחץ ☆ על כרטיס משחק."
                    : "לא נמצאו משחקים — נסה קטגוריה אחרת."}
              </p>
            ) : (
              <div className="games-panel-grid">
                {displayGames.map((g) => (
                  <GameCard
                    key={g.id}
                    game={g}
                    onPlay={onPlay}
                    isFavorite={favoriteIds.has(g.id)}
                    onToggleFavorite={(game) => void handleToggleFavorite(game)}
                    isBlacklisted={blacklistIds.has(g.id)}
                    onToggleBlacklist={(game) => void handleToggleBlacklist(game)}
                  />
                ))}
              </div>
            )}
            {isCategoryBrowse && !offline && displayGames.length > 0 && canLoadMore ? (
              <div className="games-panel-load-more-wrap">
                <button
                  type="button"
                  className="games-panel-load-more"
                  disabled={loadingMore || displayLoading}
                  onClick={() => void loadMoreBrowse()}
                >
                  {loadingMore ? "טוען עוד…" : `טען עוד ${GAMES_CATALOG_PAGE_SIZE} משחקים`}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
