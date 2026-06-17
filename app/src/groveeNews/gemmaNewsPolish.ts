import { cleanRephrasedBrief } from "./articleSummaryDisplay";
import { normalizeArticleBody } from "./engine/extract/normalizeArticleBody";
import { isLikelyInLanguage } from "./engine/summarize/languageDetect";

/** Max chars from article body sent to Gemma (full paragraphs, not Qwen notes). */
export const GEMMA_ARTICLE_EXCERPT_CHARS = 7_500;

/** Delimiters for Gemma article polish — same shape as manual Ruby chat tests. */
export const GEMMA_NEWS_PROMPT_DELIM = "|!@!@!@";
export const GEMMA_NEWS_PROMPT_END = "!@!@!@";
export const GEMMA_NEWS_INSTRUCTION = "נסח בעברית";

export function buildArticleExcerptForGemma(body: string, maxChars = GEMMA_ARTICLE_EXCERPT_CHARS): string {
  const { body: cleaned } = normalizeArticleBody(body);
  const text = cleaned.trim();
  if (!text) return "";

  const paragraphs = text
    .split(/\n{2,}/)
    .map((p) => p.replace(/\s+/g, " ").trim())
    .filter((p) => p.length > 35 && !/^title:/i.test(p));

  let out = "";
  for (const p of paragraphs) {
    const next = out ? `${out}\n\n${p}` : p;
    if (next.length > maxChars) {
      if (!out) out = p.slice(0, maxChars);
      break;
    }
    out = next;
  }

  if (!out) return text.slice(0, maxChars);
  return out.length > maxChars ? out.slice(0, maxChars) : out;
}

/** Minimal system — instruction lives in the delimited user prompt. */
export const GEMMA_NEWS_POLISH_SYSTEM =
  "ענה בעברית בלבד. שמות מותגים ומוצרים באנגלית לטינית (SpaceX, Starlink, Falcon 9). אל תחזיר את הטקסט המקורי.";

export const GEMMA_SUMMARY_FALLBACK_HE =
  "לא הצלחתי לסכם את הכתבה בעברית. ודא שהמודל נטען (טען מודל לדפדפן) ונסה שוב.";

export function buildGemmaNewsPolishUserPrompt(articleExcerpt: string, _articleTitle?: string): string {
  const excerpt = articleExcerpt.trim();
  return `${GEMMA_NEWS_INSTRUCTION}${GEMMA_NEWS_PROMPT_DELIM}
${excerpt}
${GEMMA_NEWS_PROMPT_END}`;
}

/** Strip Cyrillic/Arabic script leaks and prompt echoes from Hebrew news copy. */
export function sanitizeHebrewNewsOutput(text: string): string {
  let t = text.replace(/\r\n/g, "\n").trim();
  t = t.replace(/\|!@!@!@|!@!@!@/g, "");
  t = t.replace(/\n(?:חוקים|Rules)\s*:[\s\S]*/gi, "");
  t = t.replace(/(?:ракетת|רקטת|רקטה)\s*(?:של\s+)?(?:פל\s*קון|Falcon)\s*9?/gi, "Falcon 9");
  t = t.replace(/ת\s*פל\s*קון\s*9/gi, "Falcon 9");
  t = t.replace(/[\u0400-\u04FF]+/g, "");
  t = t.replace(/ספייס\s*[\u0600-\u06FF\w]*/gi, "SpaceX");
  t = t.replace(/\bפל\s*קון\s*9\b/gi, "Falcon 9");
  t = t.replace(/\bפל\s*קון\b/gi, "Falcon");
  t = t.replace(/(?<=[\u0590-\u05FF])[\u0600-\u06FF]{1,8}(?=[\u0590-\u05FF])/g, "");
  t = t.replace(/[ \t]{2,}/g, " ");
  t = t.replace(/\n{3,}/g, "\n\n").trim();
  return t;
}

/** True when Gemma echoed English or returned nothing usable. */
export function isFailedEnglishGemmaSummary(text: string): boolean {
  const t = text.trim();
  if (!t) return true;
  const bodyOnly = t.replace(/^(?:כותרת|תקציר)\s*:\s*/gim, " ").replace(/\s+/g, " ").trim();
  const sample = bodyOnly.slice(0, 600);
  if (!sample) return true;
  return !isLikelyInLanguage(sample, "he");
}

export function finalizeGemmaNewsSummary(polished: string | null | undefined): string {
  if (!polished?.trim()) return GEMMA_SUMMARY_FALLBACK_HE;
  const cleaned = cleanGemmaNewsPolishOutput(polished);
  if (!cleaned.trim() || isFailedEnglishGemmaSummary(cleaned)) return GEMMA_SUMMARY_FALLBACK_HE;
  return cleaned;
}

const PROMPT_LEAK_PATTERNS = [
  /\(One Clear Headline Line\)/gi,
  /\(Two or Three Short Fluent Sentences\)/gi,
  /^Rules:\s*$/gim,
  /^חוקים:\s*$/gim,
  /^-\s*No Bullet Points.*$/gim,
  /^-\s*No Numbers.*$/gim,
  /^-\s*Hebrew only.*$/gim,
  /^-\s*No English.*$/gim,
  /^-\s*Do not repeat.*$/gim,
];

/** Strip Gemma/Qwen prompt echoes and normalize to כותרת/תקציר. */
export function cleanGemmaNewsPolishOutput(raw: string): string {
  let text = raw.replace(/\r\n/g, "\n").trim();
  for (const pattern of PROMPT_LEAK_PATTERNS) {
    text = text.replace(pattern, " ");
  }
  text = sanitizeHebrewNewsOutput(text);
  text = text.replace(/\n{3,}/g, "\n\n").trim();
  return cleanRephrasedBrief(text, "he");
}

