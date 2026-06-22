import { buildLiveDisastersPayload, refreshLiveDisastersInPayload } from "./liveDisastersHits";
import { buildLivePanelShipHits, refreshLiveShipsInPayload, shipsPanelTotal } from "./liveShipsHits";
import { isNewsQuery } from "../webSearch/intents";
import { sanitizeSearchQuery } from "../webSearch/queryExtract";
import { buildWebTopicSearchPlan, planToWebSearchHint } from "../webSearch/webTopicQueryPlan";
import type { WebSearchPlanHint } from "../webSearch/types";
import type { SearchResultsPayload } from "./types";

/** Explicit SERP-panel search — focused engine query for open-web topics. */
export function buildPanelSearchPlan(query: string): WebSearchPlanHint {
  const q = sanitizeSearchQuery(query);
  const topicPlan = buildWebTopicSearchPlan(q);
  if (topicPlan) return planToWebSearchHint(topicPlan);
  return {
    queries: [q],
    answerShape: isNewsQuery(q) ? "bullet_list" : "overview",
    useWebFallback: true,
    blendNewsWithWeb: true,
  };
}

/** Payload for opening search panel — earthquakes, disasters, and cached live ships. */
export function createEmptySearchPayload(): SearchResultsPayload {
  const base = buildLiveDisastersPayload("");
  const shipHits = buildLivePanelShipHits();
  if (!shipHits.length) return base;

  const hits = [...base.hits, ...shipHits].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const shipsCount = shipsPanelTotal(hits);

  return {
    ...base,
    hits,
    facets: {
      ...base.facets,
      ships: shipsCount,
    },
  };
}

/** Refresh live disaster + ship hits when snapshot cache updates. */
export const refreshLivePanelPayload = (payload: SearchResultsPayload): SearchResultsPayload =>
  refreshLiveShipsInPayload(refreshLiveDisastersInPayload(payload));
