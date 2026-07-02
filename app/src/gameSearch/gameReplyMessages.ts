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
      `לחץ ▶ על כרטיס למטה כדי לשחק ישירות כאן בצ'אט.`
    );
  }

  if (resolved.browseMode && title && title !== "משחקים און־ליין") {
    return (
      `נמצאו ${n} משחקים בקטגוריית «${title}». ` +
      `לחץ ▶ על כרטיס למטה כדי לשחק ישירות כאן בצ'אט.`
    );
  }

  return (
    `נמצאו ${n} משחקים און־ליין. ` +
    `לחץ ▶ על כרטיס למטה כדי לשחק ישירות כאן בצ'אט.`
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
      `אפשר לחפש לפי קטגוריה — בחר מהרשימה למטה (ארקייד, PS1, מירוצים, מומלצים ועוד).`
    );
  }
  return (
    `לא מצאתי משחקים לבקשה הזו כרגע. ` +
    `נסה לבחור קטגוריה מהרשימה למטה — ארקייד, PS1, מירוצים, RPG ועוד.`
  );
}
