import { isTriviaOrSocialGame } from "../gameSearch/gameIntents";
import type { ChatUiLanguage } from "../ui/useUiLanguage";

export function isTriviaQuizRequest(text: string): boolean {
  return isTriviaOrSocialGame(text);
}

export function extractTriviaQuestionCount(text: string): number {
  const m = text.match(/(\d+)\s*(?:שאלות|questions?)/i);
  if (m) return Math.min(10, Math.max(1, Number.parseInt(m[1], 10)));
  return 5;
}

export function extractTriviaTopicTitle(text: string): string | undefined {
  const m = text.match(
    /(?:על|about|on|בנושא|ב(?:נושא|תחום))\s+([^,.!?\n]+?)(?:\s*,|\s*\.|$|\s+\d+\s*שאל|\s+עם)/i,
  );
  const title = m?.[1]?.trim();
  return title && title.length >= 2 ? title.slice(0, 72) : undefined;
}

/** Compact format law for SmolLM / Gemma when trivia intent is active. */
export function buildTriviaFormatInstruction(
  uiLang: ChatUiLanguage,
  questionCount = 5,
): string {
  const n = Math.min(10, Math.max(1, questionCount));
  if (uiLang === "he") {
    return [
      `מצב חידון: ${n} שאלות בדיוק.`,
      "כל שאלה: 4 תשובות בלבד — א), ב), ג), ד).",
      "פורמט קבוע:",
      "1) [שאלה]",
      "א) [תשובה]",
      "ב) [תשובה]",
      "ג) [תשובה]",
      "ד) [תשובה]",
      "2) …",
      "אל תחשוף תשובות נכונות. עברית בלבד. בלי הסברים מחוץ לשאלות.",
    ].join("\n");
  }
  return [
    `Quiz mode: exactly ${n} questions.`,
    "Each question: exactly 4 options — A), B), C), D).",
    "Fixed format:",
    "1) [Question]",
    "A) [option]",
    "B) [option]",
    "C) [option]",
    "D) [option]",
    "2) …",
    "Do NOT reveal correct answers. User language only. No extra commentary.",
  ].join("\n");
}

export const TRIVIA_FORMAT_INSTRUCTION = buildTriviaFormatInstruction("he");
