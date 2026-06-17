import { isLikelyEnglish } from "./engine/summarize/languageDetect";

/** True when article body should skip Qwen and use direct / Gemma path. */
export function shouldSkipQwenForArticle(text: string): boolean {
  const sample = text.trim().slice(0, 800);
  if (!sample) return true;
  return !isLikelyEnglish(sample);
}
