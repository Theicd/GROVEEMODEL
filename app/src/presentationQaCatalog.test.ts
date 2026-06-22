import { describe, expect, it } from "vitest";
import { BUILTIN_PRESENTATION_QUERY_COUNT, USER_PRESENTATION_QUERIES } from "./userPresentationQueries";
import { loadEffectiveQueries } from "./presentationQaQueryStore";

describe("presentation QA catalog", () => {
  it("has 80 built-in questions", () => {
    expect(USER_PRESENTATION_QUERIES.length).toBe(80);
    expect(BUILTIN_PRESENTATION_QUERY_COUNT).toBe(80);
  });

  it("includes new B20, E01, U01 series", () => {
    const ids = USER_PRESENTATION_QUERIES.map((q) => q.id);
    expect(ids).toContain("B20");
    expect(ids).toContain("B36");
    expect(ids).toContain("B37");
    expect(ids).toContain("E01");
    expect(ids).toContain("U15");
  });

  it("loadEffectiveQueries returns full builtin list by default", () => {
    expect(loadEffectiveQueries().length).toBeGreaterThanOrEqual(80);
  });
});
