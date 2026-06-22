import { describe, expect, it } from "vitest";
import { needsWebSearch } from "./intents";
import {
  AI_SEARCH_GAP_SCENARIOS,
  AI_SEARCH_SCENARIOS,
} from "./aiSearchQueryScenarios";
import { normalizeNewsEngineQuery } from "../groveeNews/newsQueryNormalize";
import { resolveSearchHandoff } from "./resolveSearchHandoff";
import { shouldUseSearchPlanner } from "./searchPlanner";

function termsMatch(handoffTerms: string[], expected: string[]): boolean {
  const blob = handoffTerms.join(" ").toLowerCase();
  return expected.every((needle) => blob.includes(needle.toLowerCase()));
}

describe("AI search routing QA — user query → handoff", () => {
  for (const spec of AI_SEARCH_SCENARIOS) {
    it(`${spec.id} [${spec.category}] ${spec.userQuery.slice(0, 48)}`, () => {
      const wantsWeb = spec.expectNeedsWebSearch !== false;
      expect(needsWebSearch(spec.userQuery)).toBe(wantsWeb);

      const handoff = resolveSearchHandoff(spec.userQuery);

      expect(handoff.routing).toBe(spec.expectRouting);

      for (const intent of spec.expectIntents) {
        expect(handoff.intents).toContain(intent);
      }

      expect(termsMatch(handoff.searchTerms, spec.expectSearchTerms)).toBe(true);

      if (spec.expectPanelMode) {
        expect(handoff.panelMode).toBe(spec.expectPanelMode);
      }

      if (spec.answerShape) {
        expect(handoff.answerShape).toBe(spec.answerShape);
      }

      if (spec.blendNewsWithWeb !== undefined) {
        expect(handoff.blendNewsWithWeb).toBe(spec.blendNewsWithWeb);
      }

      if (spec.expectRouting === "regex") {
        expect(shouldUseSearchPlanner(spec.userQuery)).toBe(false);
      }
      if (spec.expectRouting === "planner") {
        expect(shouldUseSearchPlanner(spec.userQuery)).toBe(true);
      }
    });
  }

  it("covers news topics and search panel modes", () => {
    const news = AI_SEARCH_SCENARIOS.filter((s) => s.expectPanelMode);
    expect(news.some((s) => s.expectPanelMode === "topics")).toBe(true);
    expect(news.some((s) => s.expectPanelMode === "search")).toBe(true);
  });

  it("covers routing paths regex (primary fast path)", () => {
    const routes = new Set(AI_SEARCH_SCENARIOS.map((s) => s.expectRouting));
    expect(routes.has("regex")).toBe(true);
  });

  it("has at least 40 scenarios", () => {
    expect(AI_SEARCH_SCENARIOS.length).toBeGreaterThanOrEqual(40);
  });
});

describe("AI search gap scenarios — conversational phrasing", () => {
  for (const gap of AI_SEARCH_GAP_SCENARIOS) {
    it(`${gap.id} needs planner path for: ${gap.userQuery.slice(0, 40)}`, () => {
      expect(needsWebSearch(gap.userQuery)).toBe(false);

      const working = resolveSearchHandoff(gap.workingQuery);
      expect(termsMatch(working.searchTerms, gap.expectSearchTerms)).toBe(true);
      expect(needsWebSearch(gap.workingQuery)).toBe(true);
    });
  }
});

describe("resolveSearchHandoff — compact output", () => {
  it("returns at most 3 search terms", () => {
    const h = resolveSearchHandoff("חפש חדשות בנושא בינה מלאכותית וסייבר");
    expect(h.searchTerms.length).toBeLessThanOrEqual(3);
  });

  it("normalizes Hebrew news topic to English engine term", () => {
    const h = resolveSearchHandoff("חפש חדשות על חלל");
    expect(h.searchTerms.join(" ").toLowerCase()).toMatch(/space/);
    expect(h.panelMode).toBe("search");
  });

  it("routes world overview to topics without planner", () => {
    const h = resolveSearchHandoff("מה קורה בעולם?");
    expect(h.routing).toBe("regex");
    expect(h.panelMode).toBe("topics");
    expect(h.intents).toContain("news");
  });

  it("normalizeNewsEngineQuery aligns with handoff searchTerms", () => {
    const q = "חפש חדשות בנושא פוליטיקה בישראל";
    const normalized = normalizeNewsEngineQuery(q);
    const h = resolveSearchHandoff(q);
    expect(h.searchTerms.join(" ").toLowerCase()).toContain(normalized.toLowerCase());
  });
});
