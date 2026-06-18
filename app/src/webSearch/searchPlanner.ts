import { classifySearchIntents, needsWebSearch, userRequestsSearch, isGeneralWebTopicQuery, isTopicalOverviewQuery, isBareWorldNewsQuery, isTimelyOverviewQuery, isNewsQuery } from "./intents";
import { expandCrossSourceIntents } from "./crossSourceIntents";
import { isGeneralNewsDigestQuery, isIsraelNewsQuery, isWorldHeadlineQuery } from "./queryExtract";
import { isTopicalOverviewRouting, topicalEnrichmentIntents, topicalProviderQuery } from "./topicalEnrichment";
import { hasUrlInQuery } from "./urlExtract";
import type { AnswerShape, SearchIntent } from "./types";

export type { AnswerShape };

export type SearchPlan = {
  intents: SearchIntent[];
  queries: string[];
  location?: string;
  answerShape: AnswerShape;
  useWebFallback?: boolean;
  /** RSS/news + SearXNG in parallel (overview questions). */
  blendNewsWithWeb?: boolean;
  skipFetch?: boolean;
  reason?: string;
};

const VALID_SHAPES = new Set<string>(["short_fact", "bullet_list", "overview", "count"]);

export const SEARCH_PLANNER_SYSTEM = `You are a fallback classifier for GROVEE (Hebrew-first live-data app).
Routing to structured APIs is done by rules — you do NOT choose intents or APIs.
Return ONLY valid JSON — no markdown, no explanation.
Schema:
{
  "queries": ["query1"],
  "answerShape": "short_fact"|"bullet_list"|"overview"|"count",
  "useWebFallback": false
}
Rules:
- 1 query max unless the user clearly asks for two distinct topics.
- answerShape: count for "כמה", bullet_list for lists, overview for world/topic status, short_fact otherwise.
- useWebFallback true ONLY when no structured live-data API fits (general web topics like robotics trends).
- useWebFallback false when the question is about weather, flights, ships, earthquakes, prices, news headlines, GitHub repos, etc.`;

export const buildSearchPlannerUserPrompt = (query: string, recentUserText: string[]): string => {
  const context =
    recentUserText.length > 0
      ? `Recent user messages:\n${recentUserText.slice(-3).map((t) => `- ${t}`).join("\n")}\n\n`
      : "";
  return `${context}User question:\n${query.trim()}\n\nJSON plan:`;
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

export const parseSearchPlanJson = (raw: string, fallbackQuery: string): SearchPlan | null => {
  const jsonStr = extractJsonObject(raw);
  if (!jsonStr) return null;
  try {
    const parsed = JSON.parse(jsonStr) as Record<string, unknown>;
    const queries = Array.isArray(parsed.queries)
      ? (parsed.queries as string[])
          .map((q) => String(q).trim())
          .filter((q) => q.length >= 2)
          .slice(0, 3)
      : [];
    const answerShape = VALID_SHAPES.has(String(parsed.answerShape))
      ? (String(parsed.answerShape) as AnswerShape)
      : "short_fact";
    const location =
      typeof parsed.location === "string" && parsed.location.trim()
        ? parsed.location.trim()
        : undefined;
    const useWebFallback = parsed.useWebFallback === true;

    if (!queries.length) queries.push(fallbackQuery.trim());

    // Intents always from rules — Gemma planner only sets fallback + shape
    const ruleQuery = queries[0] ?? fallbackQuery;
    const intents = classifySearchIntents(ruleQuery);

    return {
      intents,
      queries,
      location,
      answerShape,
      useWebFallback: useWebFallback && intents.length === 0,
    };
  } catch {
    return null;
  }
};

/** Regex-fast path — skip Gemma when routing is unambiguous. */
export const regexPlanForQuery = (query: string): SearchPlan | null => {
  const q = query.trim();
  if (!q) return null;

  if (hasUrlInQuery(q)) {
    return {
      intents: ["link"],
      queries: [q],
      answerShape: "overview",
    };
  }

  if (isIsraelNewsQuery(q)) {
    return {
      intents: ["news"],
      queries: [q],
      answerShape: "bullet_list",
      useWebFallback: false,
      reason: "Israel headlines: ynet · Walla · JPost · TOI · Israel Hayom · …",
    };
  }

  if (isWorldHeadlineQuery(q)) {
    return {
      intents: ["news"],
      queries: [q],
      answerShape: "bullet_list",
      useWebFallback: false,
      reason: "world headline: BBC · CNN · Guardian · Sky · NPR · DW · France24 · CBC",
    };
  }

  if (isGeneralNewsDigestQuery(q)) {
    return {
      intents: ["news"],
      queries: [q],
      answerShape: "bullet_list",
      useWebFallback: false,
      blendNewsWithWeb: false,
      reason: "news digest: international + Israeli RSS",
    };
  }

  if (isBareWorldNewsQuery(q)) {
    return {
      intents: ["news"],
      queries: [q, "world news headlines"],
      answerShape: "overview",
      useWebFallback: true,
      blendNewsWithWeb: true,
      reason: "overview: RSS + SearXNG",
    };
  }

  if (
    /מזג\s+האוויר/i.test(q) &&
    /(?:תנועה\s+(?:ה)?אווירית|תעבורה\s+(?:ה)?אווירית)/i.test(q) &&
    /(?:אזור|עולם|חריג)/i.test(q)
  ) {
    const intents = expandCrossSourceIntents(q, classifySearchIntents(q));
    return {
      intents,
      queries: [q, "מה מזג האוויר בישראל", "כמה מטוסים נמצאים כרגע מעל ישראל?"],
      answerShape: "short_fact",
      reason: "weather+aviation anomaly cross-source",
    };
  }

  if (isTopicalOverviewRouting(q)) {
    const intents = [...new Set<SearchIntent>([...topicalEnrichmentIntents(q), "news"])];
    const enriched = topicalProviderQuery(q);
    return {
      intents,
      queries: [enriched, q],
      answerShape: "overview",
      useWebFallback: true,
      blendNewsWithWeb: true,
      reason: `topical: ${intents.join("+")} + optional SearXNG`,
    };
  }

  if (
    userRequestsSearch(q) &&
    !hasUrlInQuery(q) &&
    !/מזג|weather|מטוס|aircraft|ספינ|ship|רעיד|earthquake/i.test(q)
  ) {
    const intents = [...new Set<SearchIntent>([...(isNewsQuery(q) ? ["news" as const] : []), "news", ...topicalEnrichmentIntents(q)])];
    return {
      intents,
      queries: [topicalProviderQuery(q), q],
      answerShape: "overview",
      useWebFallback: true,
      blendNewsWithWeb: true,
      reason: "explicit search: RSS + web + enrichment",
    };
  }

  const intents = classifySearchIntents(q);
  if (intents.length) {
    return {
      intents,
      queries: [q],
      answerShape: /כמה|how\s+many/i.test(q) ? "count" : "short_fact",
    };
  }

  if (/עומס\s+(?:ב)?(?:שמי|האוויר)|air\s+traffic\s+(?:over|above)|traffic\s+above/i.test(q)) {
    return {
      intents: ["aviation"],
      queries: [q, "כמה מטוסים מעל ישראל"],
      answerShape: "count",
      reason: "aviation synonym",
    };
  }

  if (/שמי\s+ישראל|israeli\s+airspace|מעל\s+ישראל/i.test(q) && /(?:מטוס|aircraft|עומס|traffic)/i.test(q)) {
    return {
      intents: ["aviation"],
      queries: [q],
      location: "ישראל",
      answerShape: "count",
    };
  }

  if (
    /מה\s+קור(?:ה|ה)\s+(?:ב)?(?:עולם|ישראל)|current\s+events/i.test(q) &&
    !/בינה\s+מלאכותית|\bai\b|רובוט|robotics/i.test(q) &&
    !isGeneralWebTopicQuery(q)
  ) {
    return {
      intents: ["news"],
      queries: [q, "world news headlines"],
      answerShape: "bullet_list",
      useWebFallback: true,
      blendNewsWithWeb: true,
      reason: "overview: RSS + SearXNG",
    };
  }

  if (isTopicalOverviewQuery(q)) {
    return {
      intents: [],
      queries: [q],
      answerShape: "overview",
      useWebFallback: true,
      reason: "topical: SearXNG",
    };
  }

  if (isTimelyOverviewQuery(q)) {
    const intents = topicalEnrichmentIntents(q);
    return {
      intents,
      queries: [topicalProviderQuery(q), q],
      answerShape: "overview",
      useWebFallback: true,
      blendNewsWithWeb: intents.length > 0,
      reason: "timely: multi-source + optional SearXNG",
    };
  }

  return null;
};

export const shouldUseSearchPlanner = (query: string): boolean => {
  const q = query.trim();
  if (!q || !needsWebSearch(q)) return false;
  if (regexPlanForQuery(q)) return false;

  const intents = classifySearchIntents(q);
  if (intents.length === 0) return true;

  if (/עומס|traffic|שמי\s|airspace/i.test(q) && !intents.includes("aviation")) return true;
  if (/מה\s+קור(?:ה|ה)/i.test(q) && !intents.some((i) => ["news", "hackernews", "disaster", "alerts"].includes(i))) {
    return true;
  }

  return false;
};

export const mergeSearchResults = (
  plans: SearchPlan[],
  primaryQuery: string,
): SearchPlan => {
  if (plans.length <= 1) {
    return plans[0] ?? regexPlanForQuery(primaryQuery) ?? {
      intents: classifySearchIntents(primaryQuery),
      queries: [primaryQuery],
      answerShape: "short_fact",
    };
  }
  const allIntents = new Set<SearchIntent>();
  const allQueries: string[] = [];
  let answerShape: AnswerShape = "short_fact";
  let useWebFallback = false;
  for (const p of plans) {
    p.intents.forEach((i) => allIntents.add(i));
    for (const q of p.queries) {
      if (!allQueries.includes(q)) allQueries.push(q);
    }
    if (p.answerShape === "overview") answerShape = "overview";
    if (p.useWebFallback) useWebFallback = true;
  }
  return {
    intents: [...allIntents],
    queries: allQueries.slice(0, 3),
    answerShape,
    useWebFallback,
  };
};

export const planWantsWebFallback = (
  plan: { useWebFallback?: boolean; intents?: SearchIntent[] } | undefined,
  query: string,
): boolean => {
  if (plan?.intents?.length) return false;
  if (plan?.useWebFallback) return true;
  if (userRequestsSearch(query) && !(plan?.intents?.length)) return true;
  return false;
};
