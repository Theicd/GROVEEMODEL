import { describe, expect, it } from "vitest";
import { buildBilingualSearchTerms, extractHebrewQueryTokens } from "./hebrewSearchTerms";

describe("hebrewSearchTerms", () => {
  it("extracts Hebrew tokens from query", () => {
    expect(extractHebrewQueryTokens("חדשות על איראן")).toContain("איראן");
  });

  it("maps Hebrew topic to English and Hebrew aliases", () => {
    const terms = buildBilingualSearchTerms("חדשות על איראן");
    expect(terms).toContain("iran");
    expect(terms).toContain("איראן");
  });

  it("maps israel query to Hebrew headline terms", () => {
    const terms = buildBilingualSearchTerms("חדשות ישראל");
    expect(terms).toContain("israel");
    expect(terms).toContain("ישראל");
  });
});
