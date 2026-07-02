import type { RackModelEntry } from "./modelRack/modelRack";

export type ChatModelAvailability = "gemma" | "local-text" | "none";

export const CAPABILITIES_ONLY_BANNER_HE =
  "אין מודל שיחה פעיל — חיפוש, משחקים, רדיו, טלוויזיה, התרעות ומפות עדיין זמינים. ליצירת תמונה כתוב בשיחה: «צור תמונה».";

export function buildCapabilitiesWelcomeMessage(failureReason?: string | null): string {
  const features =
    "חיפוש, משחקים, רדיו, טלוויזיה, התרעות בזמן אמת ומפות — הכול זמין. אפשר גם ליצור תמונות בשיחה.";
  const tail = "ללא מודל שיחה.";
  const reason = failureReason?.trim();
  if (reason) {
    return `ברוך הבא ל-GROVEE — ${features} (${reason}) · ${tail}`;
  }
  return `ברוך הבא ל-GROVEE — ${features} · ${tail}`;
}

export function buildCapabilitiesOnlyFallbackMessage(failureHint?: string): string {
  const lead = failureHint?.trim()
    ? `לא הצלחנו לטעון מודל שיחה במכשיר זה (${failureHint.trim()}).`
    : "אין מודל שיחה פעיל במכשיר זה.";

  return `${lead}

עדיין אפשר להשתמש ב:
• חיפוש משחקים, חדשות ומפות חיות
• ערוצי טלוויזיה ורדיו חיים
• התרעות בזמן אמת (רעידות אדמה, ספינות ועוד)
• יצירת תמונות — כתוב «תאר לי …» ואז «צור מזה תמונה»

נסה למשל: «משחקי ארקייד», «תחנות רדיו», «מפה של תל אביב», או «צור תמונה».`;
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
