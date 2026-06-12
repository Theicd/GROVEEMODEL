import { GameCard } from "./GameCard";
import type { OnlineGame } from "./gameSearch/types";

type Props = {
  games: OnlineGame[];
  onPlay: (game: OnlineGame) => void;
  onOpenPanel: () => void;
};

export function GameResultsStrip({ games, onPlay, onOpenPanel }: Props) {
  if (!games.length) return null;
  return (
    <div className="game-results-strip" dir="rtl">
      <div className="game-results-head">
        <span>🎮 משחקים און־ליין ({games.length})</span>
        <button type="button" className="game-results-more" onClick={onOpenPanel}>
          עוד משחקים →
        </button>
      </div>
      <div className="game-results-grid">
        {games.slice(0, 4).map((g) => (
          <GameCard key={g.id} game={g} compact onPlay={onPlay} />
        ))}
      </div>
    </div>
  );
}
