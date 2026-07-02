/**
 * Fix common Hebrew typos before the HE→EN translation bridge.
 * Prevents mistranslations (e.g. עבדוה → slavery instead of work).
 */

const WORD_REPLACEMENTS: ReadonlyArray<[RegExp, string]> = [
  [/לעבדוה/g, "לעבודה"],
  [/בעבדוה/g, "בעבודה"],
  [/מעבדוה/g, "מעבודה"],
  [/עבדוה/g, "עבודה"],
  [/לעבדה(?![\u0590-\u05FF])/g, "לעבודה"],
];

/** Expand vague follow-ups so translation keeps thread context. */
const PHRASE_REPLACEMENTS: ReadonlyArray<[RegExp, string]> = [
  [/^(וזה|זה)\s+(ה)?סיב(?:ה)?\s+(ש)?/i, "בגלל זה "],
  [/^(למה|מדוע)\s+(אני|אנחנו)\s+מאחר(?:ים|ת)?\s*$/i, "למה אני מאחר לעבודה"],
];

export function normalizeHebrewChatText(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return text;

  let out = trimmed;
  for (const [pattern, replacement] of WORD_REPLACEMENTS) {
    out = out.replace(pattern, replacement);
  }
  for (const [pattern, replacement] of PHRASE_REPLACEMENTS) {
    if (pattern.test(out)) {
      out = out.replace(pattern, replacement);
    }
  }
  return out;
}

export function normalizeHebrewIfNeeded(text: string, uiLang: "he" | "en"): string {
  if (uiLang !== "he") return text;
  const hasHebrew = /[\u0590-\u05FF]/.test(text);
  if (!hasHebrew) return text;
  return normalizeHebrewChatText(text);
}
