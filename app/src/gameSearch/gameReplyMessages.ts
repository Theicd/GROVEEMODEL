import type { ResolvedGameSearch } from "./types";

/** Fixed Hebrew copy when game cards are shown — avoids LLM hallucinations. */
export function buildGameSearchFoundReply(
  count: number,
  resolved: Pick<ResolvedGameSearch, "panelTitle" | "category" | "browseMode" | "query">,
): string {
  const n = count;
  const title = resolved.panelTitle.trim();
  const hasSpecificTitle = Boolean(resolved.query.trim()) && !resolved.browseMode;

  if (hasSpecificTitle && title && !title.startsWith("משחקים און")) {
    return (
      `נמצאו ${n} משחקים התואמים ל«${resolved.query.trim()}». ` +
      `הם מוצגים בחלון המשחקים שנפתח בצד ימין — בחר כרטיס ולחץ ▶ שחק עכשיו.`
    );
  }

  if (resolved.browseMode && title && title !== "משחקים און־ליין") {
    return (
      `נמצאו ${n} משחקים בקטגוריית «${title}». ` +
      `הם מוצגים בחלון המשחקים בצד ימין — בחר משחק ולחץ ▶ שחק עכשיו.`
    );
  }

  return (
    `נמצאו ${n} משחקים און־ליין. ` +
    `הם מוצגים בחלון המידע שנפתח בצד ימין של המסך — בחר כרטיס ולחץ ▶ שחק עכשיו.`
  );
}

/** Fixed Hebrew when no matching games — pairs with GameCategoryPicker in chat. */
export function buildGameSearchNotFoundReply(
  resolved: Pick<ResolvedGameSearch, "panelTitle" | "query">,
): string {
  const q = resolved.query.trim();
  if (q) {
    return (
      `לא מצאתי משחק שתואם ל«${q}». ` +
      `אפשר לחפש לפי קטגוריה — בחר מהרשימה למטה (ארקייד, PS1, מירוצים, מומלצים ועוד). ` +
      `אותן קטגוריות מופיעות גם בחלון המשחקים בצד ימין.`
    );
  }
  return (
    `לא מצאתי משחקים לבקשה הזו כרגע. ` +
    `נסה לבחור קטגוריה מהרשימה למטה — ארקייד, PS1, מירוצים, RPG ועוד — ` +
    `או מהטאbs בחלון המשחקים מצד ימין.`
  );
}
