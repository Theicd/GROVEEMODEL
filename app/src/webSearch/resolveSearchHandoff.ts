import { isTopicsOverviewQuery } from "../groveeNews/headlineIntent";
import { normalizeNewsEngineQuery } from "../groveeNews/newsQueryNormalize";
import { classifySearchIntents } from "./intents";
import { regexPlanForQuery, shouldUseSearchPlanner } from "./searchPlanner";
import type { AnswerShape, SearchIntent } from "./types";

export type SearchRouting = "regex" | "planner" | "intent-only";

/** Compact handoff from user chat → search engines (no LLM prose). */
export type SearchHandoff = {
  routing: SearchRouting;
  intents: SearchIntent[];
  /** Minimal English/keyword terms for RSS, SearXNG, GitHub, etc. */
  searchTerms: string[];
  panelMode?: "topics" | "search";
  answerShape?: AnswerShape;
  blendNewsWithWeb?: boolean;
  useWebFallback?: boolean;
};

function newsPanelMode(query: string, intents: SearchIntent[]): "topics" | "search" | undefined {
  if (!intents.includes("news")) return undefined;
  return isTopicsOverviewQuery(query) ? "topics" : "search";
}

function enrichNewsTerms(queries: string[]): string[] {
  const out: string[] = [];
  for (const line of queries) {
    const normalized = normalizeNewsEngineQuery(line);
    if (normalized) out.push(normalized);
    out.push(line);
  }
  return [...new Set(out.map((t) => t.trim()).filter((t) => t.length >= 2))].slice(0, 3);
}

/** Rule-based handoff — fast path before optional Gemma planner JSON. */
export function resolveSearchHandoff(query: string): SearchHandoff {
  const q = query.trim();
  const regex = regexPlanForQuery(q);

  if (regex) {
    const searchTerms = regex.intents.includes("news")
      ? enrichNewsTerms(regex.queries)
      : [...new Set(regex.queries.map((line) => line.trim()).filter(Boolean))].slice(0, 3);

    return {
      routing: "regex",
      intents: regex.intents,
      searchTerms,
      panelMode: newsPanelMode(q, regex.intents),
      answerShape: regex.answerShape,
      blendNewsWithWeb: regex.blendNewsWithWeb,
      useWebFallback: regex.useWebFallback,
    };
  }

  const intents = classifySearchIntents(q);
  const needsPlanner = shouldUseSearchPlanner(q);

  if (intents.length && !needsPlanner) {
    const searchTerms = intents.includes("news") ? enrichNewsTerms([q]) : [q];
    return {
      routing: "intent-only",
      intents,
      searchTerms,
      panelMode: newsPanelMode(q, intents),
      answerShape: /כמה|how\s+many/i.test(q) ? "count" : "short_fact",
    };
  }

  return {
    routing: "planner",
    intents,
    searchTerms: intents.includes("news") ? enrichNewsTerms([q]) : [q],
    panelMode: newsPanelMode(q, intents),
    answerShape: "short_fact",
    useWebFallback: intents.length === 0,
  };
}
