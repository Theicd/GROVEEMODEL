import { describe, expect, it } from "vitest";
import { classifySearchIntents } from "../webSearch/intents";
import { needsWebSearch } from "../webSearch/intents";
import { isTopicsOverviewQuery } from "./headlineIntent";
import {
  isExplicitNewsTopicSearch,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import { NEWS_TOPIC_ACCEPTANCE_QUERIES } from "./newsTopicAcceptanceQueries";

describe("NEWS topic queries — routing + normalize (25 queries)", () => {
  for (const q of NEWS_TOPIC_ACCEPTANCE_QUERIES) {
    it(`${q.id}: routes to news search — ${q.query.slice(0, 36)}`, () => {
      expect(needsWebSearch(q.query)).toBe(true);
      expect(isExplicitNewsTopicSearch(q.query)).toBe(true);
      expect(isSpecificNewsTopicQuery(q.query)).toBe(true);
      expect(isTopicsOverviewQuery(q.query)).toBe(false);
      for (const intent of q.expectIntents) {
        expect(classifySearchIntents(q.query)).toContain(intent);
      }
      expect(isTopicsOverviewQuery(q.query) ? "topics" : "search").toBe(q.expectPanelMode);
    });

    it(`${q.id}: engine query — ${q.expectEngineQuery}`, () => {
      const normalized = normalizeNewsEngineQuery(q.query);
      expect(normalized.toLowerCase()).toContain(q.expectEngineQuery.toLowerCase().split(" ")[0]);
      for (const token of q.expectEngineQuery.toLowerCase().split(/\s+/)) {
        expect(normalized.toLowerCase()).toContain(token);
      }
    });
  }

  it("has 25 topic queries", () => {
    expect(NEWS_TOPIC_ACCEPTANCE_QUERIES).toHaveLength(25);
  });
});
