import { describe, expect, it } from "vitest";

import { buildArchiveSearchQueries } from "./internetArchiveSearch";

describe("internetArchiveSearch", () => {
  it("builds general and Hebrew archive queries", () => {
    const queries = buildArchiveSearchQueries("סרט ישן", "סרט ישן");
    expect(queries.length).toBeGreaterThanOrEqual(2);
    expect(queries[0]).toContain("mediatype:movies");
    expect(queries.some((q) => q.includes("language:Hebrew"))).toBe(true);
  });

  it("adds Channel 11 / Kan clause for TV archive queries", () => {
    const queries = buildArchiveSearchQueries("ערוץ 11 תוכנית", "ערוץ 11 תוכנית");
    expect(queries.some((q) => /channel 11|כאן/i.test(q))).toBe(true);
  });
});
