/**
 * Gemma / planner hook — short Hebrew search phrases for open-web topics.
 * Rule-based defaults live in webTopicQueryPlan.ts; planner refines when model is available.
 */
import { sanitizeSearchQuery } from "./queryExtract";

export const OPEN_WEB_QUERY_PLANNER_SYSTEM = `You generate SHORT Hebrew search phrases for a web search engine.
Return ONLY valid JSON — no markdown.
Schema: {"queries":["phrase1","phrase2","phrase3"]}

Rules:
- 2-3 phrases, each 3-8 Hebrew words ONLY.
- NO English words. NO site: operators. NO URLs. NO year unless user asked.
- NO full user sentence. NO placeholders.
- Match the user's topic exactly (cinema / sports / events / F1).

Examples:
User: סרטים בקולנוע בישראל → {"queries":["סרטים בקולנוע עכשיו","מה מציגים בקולנוע","הסרטים המצליחים בקולנוע"]}
User: מי זכה ביורו → {"queries":["מנצחת יורו 2024","שחקן מצטיין יורו","גמר יורו אלופה"]}`;

export const buildOpenWebPlannerUserPrompt = (
  userQuery: string,
  ruleQueries: string[],
): string => {
  const q = sanitizeSearchQuery(userQuery);
  return [
    `שאלת המשתמש:\n${q}`,
    "",
    `הצעות ברירת מחדל (אפשר לשפר, לא חובה להעתיק): ${ruleQueries.join(" | ")}`,
    "",
    "JSON:",
  ].join("\n");
};

const extractJsonObject = (text: string): string | null => {
  const trimmed = text.trim();
  const fence = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence?.[1]) return fence[1].trim();
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start >= 0 && end > start) return trimmed.slice(start, end + 1);
  return null;
};

const isValidEnginePhrase = (phrase: string): boolean => {
  const p = phrase.trim();
  if (p.length < 4 || p.length > 80) return false;
  if (/site:|https?:|\.com\b|\bor\b|\band\b/i.test(p)) return false;
  if (/[\[\]{}]/.test(p)) return false;
  if (!/[\u0590-\u05FF]/.test(p)) return false;
  return true;
};

/** Parse Gemma JSON; fall back to rule-based queries. */
export const parseOpenWebQueriesJson = (raw: string | null, fallback: string[]): string[] => {
  if (!raw?.trim()) return fallback;
  const jsonStr = extractJsonObject(raw);
  if (!jsonStr) return fallback;
  try {
    const parsed = JSON.parse(jsonStr) as { queries?: unknown };
    const list = Array.isArray(parsed.queries)
      ? (parsed.queries as string[])
          .map((q) => String(q).trim())
          .filter(isValidEnginePhrase)
          .slice(0, 3)
      : [];
    return list.length ? list : fallback;
  } catch {
    return fallback;
  }
};

export const hasPlaceholderReply = (text: string): boolean =>
  /\[כותרת|\[Movie|\[תקציר|\[סרט\s+\d|כותרת\s+סרט\s+\d\s+בעברית/i.test(text);

export const buildOpenWebFailureReply = (engineQueries: string[]): string =>
  [
    "לא מצאתי כרגע מידע מדויק מהאתרים — נסה שוב בעוד רגע.",
    `ביטויי חיפוש: ${engineQueries.join(" · ")}`,
    "Sources: (none)",
  ].join("\n");
