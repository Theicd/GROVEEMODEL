import type { NewsPanelMode } from "./types";

export type NewsPanelGuideOptions = {
  mode: NewsPanelMode;
  cardCount: number;
};

/** Short chat reply when the news side panel opens — no headline dump, no Gemma summary. */
export function buildNewsPanelGuideReply(
  query: string,
  options: NewsPanelGuideOptions | null,
): string {
  const count = options?.cardCount ?? 0;
  if (count <= 0) {
    return "לא נמצאו כתבות תואמות במאגר המקומי. נסה ניסוח אחר או המתן שהמאגר יתמלא.";
  }

  const modeLine =
    options!.mode === "topics"
      ? "סקירת נושאים מובילים"
      : `חיפוש: ${query.trim().slice(0, 56)}`;

  return [
    `נפתחו ${count} כרטיסיות חדשות בפאנל מימין (${modeLine}).`,
    "",
    "לחץ על כרטיסיה כדי לפתוח אותה, ואז:",
    "• «מקור ↗» — לקרוא את הכתבה המלאה באתר המקורי",
    "• «סכם כתבה» — לקבל תקציר קצר בעברית",
  ].join("\n");
}
