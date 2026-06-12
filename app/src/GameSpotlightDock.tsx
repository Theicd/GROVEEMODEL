import { useGameSpotlightPool } from "./gameSearch/useGameSpotlightPool";

type Props = {
  visible: boolean;
  onOpenPanel: () => void;
};

export function GameSpotlightDock({ visible, onOpenPanel }: Props) {
  const { games, current } = useGameSpotlightPool();

  if (!visible || !games.length || !current) return null;

  return (
    <div className="game-spotlight-dock" dir="rtl" aria-label="משחקים מומלצים">
      <button
        type="button"
        className="game-spotlight-dock-tab"
        onClick={onOpenPanel}
        aria-label="פתח משחקים מומלצים"
      >
        <span className="game-spotlight-dock-tab-glow" aria-hidden="true" />
        <div className="game-spotlight-dock-tab-preview">
          <img
            key={current.id}
            src={current.thumbnail}
            alt=""
            className="game-spotlight-dock-tab-thumb"
            loading="lazy"
          />
        </div>
        <span className="game-spotlight-dock-tab-label">משחקים</span>
        <span className="game-spotlight-dock-tab-hint">מומלצים</span>
        <span className="game-spotlight-dock-tab-icon" aria-hidden="true">
          🎮
        </span>
      </button>
    </div>
  );
}
