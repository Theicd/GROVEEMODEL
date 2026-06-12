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

type Props = {
  games: OnlineGame[];
  loading: boolean;
  embedGame: OnlineGame | null;
  title?: string;
  initialCategory?: GameCategoryId | null;
  onClose: () => void;
  onPlay: (game: OnlineGame) => void;
  onBackFromEmbed: () => void;
  onGamesUpdate: (games: OnlineGame[], title?: string) => void;
  onLoadingChange: (loading: boolean) => void;
};

export function GamesPanel({
  games,
  loading,
  embedGame,
  title = "משחקים און־ליין",
  initialCategory = null,
  onClose,
  onPlay,
  onBackFromEmbed,
  onGamesUpdate,
  onLoadingChange,
}: Props) {
  const [searchInput, setSearchInput] = useState("");
  const [expandedSearch, setExpandedSearch] = useState(false);
  const [activeCategory, setActiveCategory] = useState<GameCategoryId>(
    initialCategory ?? "featured",
  );

  useEffect(() => {
    if (initialCategory) setActiveCategory(initialCategory);
  }, [initialCategory]);

  const runSearch = useCallback(
    async (query: string, category: GameCategoryId | null = null) => {
      onLoadingChange(true);
      try {
        const resolved = parseGameUserRequest(query);
        if (category) resolved.category = category;
        const result = await searchOnlineGamesWithFallback(resolved, 16);
        onGamesUpdate(result.games, query ? `חיפוש: ${query}` : resolved.panelTitle);
      } finally {
        onLoadingChange(false);
      }
    },
    [onGamesUpdate, onLoadingChange],
  );

  const loadCategory = useCallback(
    async (cat: GameCategoryId) => {
      setActiveCategory(cat);
      onLoadingChange(true);
      try {
        const result = await randomOnlineGames(20, cat);
        onGamesUpdate(result.games, GAME_CATEGORIES.find((c) => c.id === cat)?.labelHe);
      } finally {
        onLoadingChange(false);
      }
    },
    [onGamesUpdate, onLoadingChange],
  );

  const refreshList = useCallback(async () => {
    onLoadingChange(true);
    try {
      const result = await randomOnlineGames(20, activeCategory);
      onGamesUpdate(result.games, GAME_CATEGORIES.find((c) => c.id === activeCategory)?.labelHe);
    } finally {
      onLoadingChange(false);
    }
  }, [activeCategory, onGamesUpdate, onLoadingChange]);

  return (
    <div className="games-panel-inner">
      <header className="games-panel-head">
        <div className="games-panel-title">
          <span className="games-panel-dot" aria-hidden="true" />
          {embedGame ? embedGame.title : title}
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
          {expandedSearch ? (
            <div className="games-panel-search-bar">
              <input
                type="search"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="חפש משחק… (ארקייד, pacman, shooter)"
                onKeyDown={(e) => {
                  if (e.key === "Enter") void runSearch(searchInput, activeCategory);
                }}
              />
              <button type="button" onClick={() => void runSearch(searchInput, activeCategory)}>
                חפש
              </button>
            </div>
          ) : null}

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

          <div className="games-panel-body">
            {loading ? (
              <p className="games-panel-status">טוען משחקים…</p>
            ) : games.length === 0 ? (
              <p className="games-panel-status">לא נמצאו משחקים — נסה קטגוריה אחרת.</p>
            ) : (
              <div className="games-panel-grid">
                {games.map((g) => (
                  <GameCard key={g.id} game={g} onPlay={onPlay} />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
