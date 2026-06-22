import { GAME_CATEGORIES } from "./gameSearch/archiveQueries";
import type { GameCategoryId } from "./gameSearch/types";

type Props = {
  activeCategory?: GameCategoryId | null;
  onPick: (category: GameCategoryId) => void;
  onOpenFavorites?: () => void;
  compact?: boolean;
};

export function GameCategoryPicker({ activeCategory, onPick, onOpenFavorites, compact }: Props) {
  return (
    <div
      className={`game-category-picker${compact ? " game-category-picker--compact" : ""}`}
      dir="rtl"
      role="listbox"
      aria-label="קטגוריות משחקים"
    >
      <p className="game-category-picker-lead">
        {compact
          ? "בחר קטגוריה לגלישה:"
          : "לא מצאתי את המשחק המבוקש — אפשר לגלוש לפי קטגוריות:"}
      </p>
      {onOpenFavorites ? (
        <button type="button" className="game-category-picker-favorites" onClick={onOpenFavorites}>
          ★ מועדפים — משחקים ששמרת
        </button>
      ) : null}
      <div className="game-category-picker-grid">
        {GAME_CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            type="button"
            role="option"
            aria-selected={activeCategory === cat.id}
            className={`game-category-picker-btn${activeCategory === cat.id ? " active" : ""}`}
            onClick={() => onPick(cat.id)}
          >
            <span className="game-category-picker-icon" aria-hidden="true">
              {cat.icon}
            </span>
            <span className="game-category-picker-label">{cat.labelHe}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
