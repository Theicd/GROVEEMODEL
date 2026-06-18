import { classifySearchIntents, sanitizeSearchQuery } from "./intents";
import { isSearxngConfigured } from "./providers/searxng";
import { planWantsWebFallback, regexPlanForQuery } from "./searchPlanner";
import type { AnswerShape, SearchIntent, WebSearchPlanHint } from "./types";

/** Data layer — Structured APIs, News feeds, or web search fallback only. */
export type DataTier = "structured" | "news" | "web_fallback";

export const NEWS_INTENTS: SearchIntent[] = ["news", "hackernews"];

export const tierForIntent = (intent: SearchIntent): DataTier => {
  if (NEWS_INTENTS.includes(intent)) return "news";
  return "structured";
};

export const primaryTierForIntents = (intents: SearchIntent[]): DataTier => {
  if (!intents.length) return "web_fallback";
  if (intents.some((i) => !NEWS_INTENTS.includes(i))) return "structured";
  return "news";
};

export type QueryRoute = {
  intents: SearchIntent[];
  tier: DataTier;
  answerShape: AnswerShape;
  useWebFallback: boolean;
  blendNewsWithWeb: boolean;
  queries: string[];
  fromRegex: boolean;
};

/**
 * Central query routing — rules/regex only; Gemma planner may set useWebFallback
 * and answerShape but never overrides rule-based intents.
 */
export const routeQuery = (query: string, planHint?: WebSearchPlanHint): QueryRoute => {
  const q = sanitizeSearchQuery(query);
  const regexPlan = regexPlanForQuery(q);
  const ruleIntents = classifySearchIntents(q);
  const intents = regexPlan ? regexPlan.intents : ruleIntents;

  const answerShape: AnswerShape =
    planHint?.answerShape ??
    regexPlan?.answerShape ??
    (/כמה|how\s+many/i.test(q) ? "count" : "short_fact");

  const queries =
    planHint?.queries?.length && planHint.queries.length > 1
      ? planHint.queries.slice(0, 3)
      : regexPlan?.queries?.length && regexPlan.queries.length > 1
        ? regexPlan.queries.slice(0, 3)
        : [q];

  const hasStructuredTasks = intents.length > 0;
  const blendNewsWithWeb = regexPlan?.blendNewsWithWeb === true || planHint?.blendNewsWithWeb === true;
  const useWebFallback =
    blendNewsWithWeb ||
    (!hasStructuredTasks &&
      (regexPlan?.useWebFallback === true ||
        planHint?.useWebFallback === true ||
        planWantsWebFallback({ useWebFallback: planHint?.useWebFallback, intents: [] }, q)));

  return {
    intents,
    tier: blendNewsWithWeb ? "news" : useWebFallback && !hasStructuredTasks ? "web_fallback" : primaryTierForIntents(intents),
    answerShape,
    useWebFallback,
    blendNewsWithWeb,
    queries,
    fromRegex: !!regexPlan,
  };
};

/** Critical rule: structured API → no SearXNG — except overview blend (RSS + web). */
export const shouldAllowWebFallback = (
  structuredTaskCount: number,
  plan?: { useWebFallback?: boolean; blendNewsWithWeb?: boolean },
  query = "",
): boolean => {
  if (plan?.blendNewsWithWeb) return true;
  if (plan?.useWebFallback) return true;
  if (structuredTaskCount > 0) return false;
  if (!isSearxngConfigured()) return false;
  return planWantsWebFallback(plan, query);
};
