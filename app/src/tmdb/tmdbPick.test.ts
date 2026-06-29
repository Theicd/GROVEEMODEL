import { describe, expect, it } from "vitest";
import { pickMovie, pickTv } from "./tmdbClient";

describe("tmdb pickTv", () => {
  it("matches the EPG title against original_name when name is localized (Hebrew UI)", () => {
    const results = [
      { id: 1, name: "נוכלת בשירות החוק", original_name: "Wild Cards", first_air_date: "2024-01-10", vote_count: 63, popularity: 18 },
    ];
    expect(pickTv(results, "Wild Cards")?.id).toBe(1);
  });

  it("prefers the higher-voted exact match over an obscure one", () => {
    const results = [
      { id: 1, name: "Wild Cards", original_name: "Wild Cards", vote_count: 0, popularity: 0.1 },
      { id: 2, name: "Wild Cards", original_name: "Wild Cards", vote_count: 63, popularity: 18 },
    ];
    expect(pickTv(results, "Wild Cards")?.id).toBe(2);
  });
});

describe("tmdb pickMovie", () => {
  it("rejects an obscure 0-vote partial match (Wild Cards documentary)", () => {
    const results = [
      {
        id: 1,
        title: "Wild Cards - The Artistry Of Playing Cards",
        original_title: "Wild Cards - The Artistry Of Playing Cards",
        release_date: "2016-06-16",
        vote_count: 0,
        popularity: 0.1,
      },
    ];
    expect(pickMovie(results, "Wild Cards")).toBeNull();
  });

  it("still matches an exact movie title", () => {
    const results = [
      { id: 9, title: "Step Brothers", original_title: "Step Brothers", release_date: "2008-07-25", vote_count: 5000, popularity: 30 },
    ];
    expect(pickMovie(results, "Step Brothers")?.id).toBe(9);
  });
});
