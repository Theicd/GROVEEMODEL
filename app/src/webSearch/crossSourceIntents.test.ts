import { describe, expect, it } from "vitest";
import { expandCrossSourceIntents, isCrossSourceQuery } from "./crossSourceIntents";
import { classifySearchIntents, needsWebSearch } from "./intents";

describe("crossSourceIntents", () => {
  it("detects cross-source Hebrew patterns", () => {
    expect(isCrossSourceQuery("האם יש כרגע מטוסים מעל אזור שבו יש סופה פעילה?")).toBe(true);
    expect(isCrossSourceQuery("מה מזג האוויר בתל אביב")).toBe(false);
  });

  it("expands intents for storm + aviation", () => {
    const query = "האם יש כרגע מטוסים מעל אזור שבו יש סופה פעילה?";
    const intents = expandCrossSourceIntents(query, []);
    expect(intents).toContain("aviation");
    expect(intents).toContain("disaster");
  });

  it("classifySearchIntents adds multiple providers for C01", () => {
    const query = "האם יש כרגע מטוסים מעל אזור שבו יש סופה פעילה?";
    const intents = classifySearchIntents(query);
    expect(intents.length).toBeGreaterThanOrEqual(2);
    expect(needsWebSearch(query)).toBe(true);
  });

  it("AWACS triggers search and aviation intent", () => {
    const query = "כמה מטוסי AWACS פעילים כרגע?";
    expect(needsWebSearch(query)).toBe(true);
    expect(classifySearchIntents(query)).toContain("aviation");
  });

  it("simple aircraft count over Israel is not cross-source", () => {
    const query = "כמה מטוסים נמצאים כרגע מעל ישראל?";
    expect(isCrossSourceQuery(query)).toBe(false);
    const intents = classifySearchIntents(query);
    expect(intents).toContain("aviation");
    expect(intents).not.toContain("marine");
    expect(intents).not.toContain("alerts");
  });
});
