import { describe, expect, it } from "vitest";
import { classifySearchIntents } from "../webSearch/intents";
import { needsWebSearch } from "../webSearch/intents";
import { isTopicsOverviewQuery } from "./headlineIntent";
import { normalizeNewsEngineQuery } from "./newsQueryNormalize";
import { NEWS_ACCEPTANCE_QUERIES } from "./newsAcceptanceQueries";

describe("NEWS chat acceptance — routing (10 queries)", () => {
  for (const q of NEWS_ACCEPTANCE_QUERIES) {
    it(`${q.id} intents + panel mode: ${q.query.slice(0, 40)}`, () => {
      expect(needsWebSearch(q.query)).toBe(true);
      for (const intent of q.expectIntents) {
        expect(classifySearchIntents(q.query)).toContain(intent);
      }

      const topics = isTopicsOverviewQuery(q.query);
      if (q.expectPanelMode === "topics") {
        expect(topics).toBe(true);
      } else {
        expect(topics).toBe(false);
      }
    });

    if (q.expectEngineQuery) {
      it(`${q.id} engine query normalize`, () => {
        const normalized = normalizeNewsEngineQuery(q.query);
        expect(normalized.toLowerCase()).toContain(q.expectEngineQuery!.toLowerCase());
      });
    }
  }

  it("covers both panel modes", () => {
    const modes = new Set(NEWS_ACCEPTANCE_QUERIES.map((q) => q.expectPanelMode));
    expect(modes.has("topics")).toBe(true);
    expect(modes.has("search")).toBe(true);
  });

  it("has exactly 11 queries", () => {
    expect(NEWS_ACCEPTANCE_QUERIES).toHaveLength(11);
  });
});
