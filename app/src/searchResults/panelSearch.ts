import { isNewsQuery } from "../webSearch/intents";
import type { WebSearchPlanHint } from "../webSearch/types";
import type { SearchResultsPayload } from "./types";

/** Explicit SERP-panel search — always blend RSS + web + enrichment providers. */
export function buildPanelSearchPlan(query: string): WebSearchPlanHint {
  const q = query.trim();
  return {
    queries: [q],
    answerShape: isNewsQuery(q) ? "bullet_list" : "overview",
    useWebFallback: true,
    blendNewsWithWeb: true,
  };
}

/** Empty payload for opening the search panel from the sidebar. */
export function createEmptySearchPayload(): SearchResultsPayload {
  return {
    query: "",
    generatedAt: Date.now(),
    hits: [],
    facets: { rss: 0, web: 0, repos: 0, papers: 0, movies: 0, images: 0, videos: 0, youtube: 0, products: 0, hfModels: 0, other: 0 },
    providerErrors: [],
  };
}
