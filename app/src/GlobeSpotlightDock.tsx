import { GlobeVisual } from "./GlobeVisual";

type Props = {
  visible: boolean;
  onOpenPanel: () => void;
};

/** Same footprint as GameSpotlightDock — cyan theme only. */
export function GlobeSpotlightDock({ visible, onOpenPanel }: Props) {
  if (!visible) return null;

  return (
    <div className="game-spotlight-dock globe-spotlight-dock" dir="rtl" aria-label="עולם חי">
      <button
        type="button"
        className="game-spotlight-dock-tab globe-spotlight-dock-tab--globe"
        onClick={onOpenPanel}
        aria-label="פתח מוניטור עולם חי"
      >
        <span className="game-spotlight-dock-tab-glow" aria-hidden="true" />
        <div className="game-spotlight-dock-tab-preview globe-spotlight-dock-tab-preview">
          <GlobeVisual size="sm" pulse />
        </div>
        <span className="game-spotlight-dock-tab-label">עולם</span>
        <span className="game-spotlight-dock-tab-hint">חי</span>
        <span className="game-spotlight-dock-tab-icon" aria-hidden="true">
          🌐
        </span>
      </button>
    </div>
  );
}
