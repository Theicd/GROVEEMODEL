import type { RackModelEntry } from "./modelRack/modelRack";

export type ChatModelAvailability = "gemma" | "local-text" | "none";

export const CAPABILITIES_ONLY_BANNER_HE =
  "אין מודל שיחה פעיל — חיפוש, משחקים, רדיו, טלוויזיה, התרעות ומפות עדיין זמינים. ליצירת תמונה בחר מודל תמונה בבורר.";

export function buildCapabilitiesOnlyFallbackMessage(failureHint?: string): string {
  const lead = failureHint?.trim()
    ? `לא הצלחנו לטעון מודל שיחה במכשיר זה (${failureHint.trim()}).`
    : "אין מודל שיחה פעיל במכשיר זה.";

  return `${lead}

עדיין אפשר להשתמש ב:
• חיפוש משחקים, חדשות ומפות חיות
• ערוצי טלוויזיה ורדיו חיים
• התרעות בזמן אמת (רעידות אדמה, ספינות ועוד)
• יצירת תמונות — בחר מודל תמונה בבורר למעלה

נסה למשל: «משחקי ארקייד», «תחנות רדיו», «מפה של תל אביב», או תאר תמונה אחרי בחירת מודל יצירה.`;
}

/** Prefer cloud image model when chat LLM is unavailable. */
export function pickCapabilitiesDefaultRackId(rack: RackModelEntry[]): string | null {
  const ready = rack.filter(
    (r) =>
      r.status === "ready" &&
      r.adapter !== "gemma-local" &&
      r.adapter !== "hf-local-text",
  );
  const image =
    ready.find((r) => r.modality === "image" && r.adapter === "pollinations") ??
    ready.find((r) => r.modality === "image") ??
    null;
  return image?.id ?? null;
}
