import { useCallback, useEffect, useState } from "react";
import { GameCard } from "./GameCard";
import type { OnlineGame } from "./gameSearch/types";
import { getFavoriteIds, toggleFavoriteGame } from "./localExperience/gamesStore";

type Props = {
  games: OnlineGame[];
  onPlay: (game: OnlineGame) => void;
  onOpenPanel: () => void;
  onOpenFavorites?: () => void;
};

export function GameResultsStrip({ games, onPlay, onOpenPanel, onOpenFavorites }: Props) {
  const [favoriteIds, setFavoriteIds] = useState<Set<string>>(new Set());

  const refreshFavorites = useCallback(async () => {
    setFavoriteIds(await getFavoriteIds());
  }, []);

  useEffect(() => {
    void refreshFavorites();
  }, [refreshFavorites]);

  const handleToggleFavorite = useCallback(
    async (game: OnlineGame) => {
      await toggleFavoriteGame(game);
      await refreshFavorites();
    },
    [refreshFavorites],
  );

  if (!games.length) return null;
  return (
    <div className="game-results-strip" dir="rtl">
      <div className="game-results-head">
        <span>🎮 משחקים און־ליין ({games.length})</span>
        <div className="game-results-head-actions">
          {onOpenFavorites ? (
            <button type="button" className="game-results-more" onClick={onOpenFavorites}>
              ★ מועדפים
            </button>
          ) : null}
          <button type="button" className="game-results-more" onClick={onOpenPanel}>
            עוד משחקים →
          </button>
        </div>
      </div>
      <div className="game-results-grid">
        {games.slice(0, 4).map((g) => (
          <GameCard
            key={g.id}
            game={g}
            compact
            onPlay={onPlay}
            isFavorite={favoriteIds.has(g.id)}
            onToggleFavorite={(game) => void handleToggleFavorite(game)}
          />
        ))}
      </div>
    </div>
  );
}
