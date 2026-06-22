import { useCallback, useEffect, useState } from "react";
import { GameCard } from "./GameCard";
import { GAME_CATEGORIES } from "./gameSearch/archiveQueries";
import {
  parseGameUserRequest,
  randomOnlineGames,
  searchOnlineGamesWithFallback,
  type GameCategoryId,
  type OnlineGame,
} from "./gameSearch";
import {
  getFavoriteGames,
  getFavoriteIds,
  getRecentPlayedGames,
  loadGamesSession,
  toggleFavoriteGame,
} from "./localExperience/gamesStore";
import { useNetworkStatus } from "./hooks/useNetworkStatus";

type PanelView = "browse" | "recent" | "favorites";

type Props = {
  games: OnlineGame[];
  loading: boolean;
  embedGame: OnlineGame | null;
  title?: string;
  initialCategory?: GameCategoryId | null;
  startView?: PanelView;
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

  const refreshFavorites = useCallback(async () => {
    setFavoriteIds(await getFavoriteIds());
  }, []);

  useEffect(() => {
    if (initialCategory) setActiveCategory(initialCategory);
  }, [initialCategory]);

  useEffect(() => {
    void refreshFavorites();
  }, [refreshFavorites]);

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
      try {
        const result = await randomOnlineGames(20, cat);
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
    try {
      const result = await randomOnlineGames(20, activeCategory);
      onGamesUpdate(result.games, GAME_CATEGORIES.find((c) => c.id === activeCategory)?.labelHe, activeCategory);
    } finally {
      onLoadingChange(false);
    }
  }, [activeCategory, loadStoredView, offline, onGamesUpdate, onLoadingChange, panelView]);

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

  const displayGames =
    panelView === "browse" ? games : filterStoredGames(storedGames);
  const displayLoading = panelView === "browse" && loading;
  const favCount = favoriteIds.size;

  return (
    <div className="games-panel-inner">
      <header className="games-panel-head">
        <div className="games-panel-title">
          <span className="games-panel-dot" aria-hidden="true" />
          {embedGame ? embedGame.title : panelView === "recent" ? "שיחקת לאחרונה" : panelView === "favorites" ? "מועדפים" : title}
        </div>
        <div className="games-panel-head-actions">
          {embedGame ? (
            <button type="button" className="games-panel-btn" onClick={onBackFromEmbed}>
              ← רשימה
            </button>
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
          <iframe
            className="games-panel-embed"
            src={embedGame.embedUrl}
            title={embedGame.title}
            allowFullScreen
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
          />
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
        <>
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

          {panelView === "browse" ? (
            <div className="games-panel-categories" role="tablist">
              {GAME_CATEGORIES.map((cat) => (
                <button
                  key={cat.id}
                  type="button"
                  role="tab"
                  aria-selected={activeCategory === cat.id}
                  className={`games-panel-cat${activeCategory === cat.id ? " active" : ""}`}
                  onClick={() => void loadCategory(cat.id)}
                >
                  {cat.icon} {cat.labelHe}
                </button>
              ))}
            </div>
          ) : null}

          <div className="games-panel-body">
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
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
