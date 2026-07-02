/** Compact personality blocks — token-budget aware per model tier. */

export const DEFAULT_PERSONALITY_HE =
  "סגנון: חם, קצר, עוזר — כמו עוזר אישי בדפדפן. אל תאריך.";

export const DEFAULT_PERSONALITY_EN =
  "Style: warm, brief, helpful — like a browser assistant. Do not ramble.";

const SMOL_MAX = 120;
const GEMMA_MAX = 320;

function clip(s: string, max: number): string {
  const t = s.trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

export function personalityBlockForSmolLM(uiLang: "he" | "en"): string {
  const base = uiLang === "he" ? DEFAULT_PERSONALITY_HE : DEFAULT_PERSONALITY_EN;
  return clip(base, SMOL_MAX);
}

export function personalityBlockForGemma(uiLang: "he" | "en"): string {
  const base = uiLang === "he" ? DEFAULT_PERSONALITY_HE : DEFAULT_PERSONALITY_EN;
  return clip(base, GEMMA_MAX);
}
