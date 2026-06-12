import type { OnlineGame } from "./gameSearch/types";
import { useGameSpotlightPool } from "./gameSearch/useGameSpotlightPool";

type Props = {
  onPlay: (game: OnlineGame) => void;
  onOpenPanel: () => void;
};

/** @deprecated use GameSpotlightDock in chat area */
export function GameSpotlight({ onPlay, onOpenPanel }: Props) {
  const { games, index, current } = useGameSpotlightPool();

  if (!games.length || !current) return null;

  return (
    <div className="game-spotlight" dir="rtl">
      <div className="game-spotlight-head">
        <span>🎮 משחקים</span>
        <button type="button" className="game-spotlight-more" onClick={onOpenPanel}>
          עוד
        </button>
      </div>
      <button
        type="button"
        className="game-spotlight-card"
        onClick={() => onPlay(current)}
        aria-label={`שחק ${current.title}`}
      >
        <img src={current.thumbnail} alt="" className="game-spotlight-img" loading="lazy" />
        <span className="game-spotlight-title">{current.title}</span>
        {current.year ? (
          <span className="game-spotlight-sub">
            {current.platform} · {current.year}
          </span>
        ) : (
          <span className="game-spotlight-sub">{current.platform}</span>
        )}
        <span className="game-spotlight-cta">▶ שחק עכשיו</span>
      </button>
      <div className="game-spotlight-dots" aria-hidden="true">
        {games.map((g, i) => (
          <span key={g.id} className={`game-spotlight-dot${i === index ? " active" : ""}`} />
        ))}
      </div>
    </div>
  );
}
