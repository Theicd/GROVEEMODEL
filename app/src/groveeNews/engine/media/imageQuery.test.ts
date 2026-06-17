import { describe, expect, it } from "vitest";
import {
  buildImageSearchQuery,
  buildStockSearchQueries,
  scoreStockCandidate,
  tokenizeForImageMatch,
} from "./imageQuery";

describe("buildImageSearchQuery", () => {
  it("drops stop words and keeps topic terms", () => {
    expect(buildImageSearchQuery("NASA Mars rover finds new rock samples", "space")).toMatch(/space/i);
    expect(buildImageSearchQuery("NASA Mars rover finds new rock samples", "space")).toMatch(/NASA|Mars|rover/i);
  });

  it("builds English-only queries from headline", () => {
    const q = buildImageSearchQuery("Champions League football final preview", "sport");
    expect(q.toLowerCase()).toContain("sport");
    expect(q.toLowerCase()).toContain("champions");
  });
});

describe("buildStockSearchQueries", () => {
  it("returns deduped variants", () => {
    const queries = buildStockSearchQueries("Tesla unveils affordable electric car", "car");
    expect(queries.length).toBeGreaterThanOrEqual(2);
    expect(queries[0].toLowerCase()).toContain("car");
  });
});

describe("scoreStockCandidate", () => {
  it("prefers tag overlap over random images", () => {
    const query = "NASA Mars rover space";
    const good = scoreStockCandidate(query, "mars rover nasa space planet red");
    const bad = scoreStockCandidate(query, "abstract wallpaper background texture");
    expect(good).toBeGreaterThan(bad);
  });

  it("tokenize removes stop words", () => {
    expect(tokenizeForImageMatch("the rocket launch")).not.toContain("the");
    expect(tokenizeForImageMatch("the rocket launch")).toContain("rocket");
  });
});
