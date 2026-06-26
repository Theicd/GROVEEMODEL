import { describe, expect, it } from "vitest";
import { shouldUseTmdbForEpg } from "./tmdbEpgGate";

describe("shouldUseTmdbForEpg", () => {
  it("skips generic sports segment titles", () => {
    expect(shouldUseTmdbForEpg({ channelId: "x", title: "Billiards", start: new Date(), end: new Date() })).toBe(
      false,
    );
  });

  it("allows episodic programmes", () => {
    expect(
      shouldUseTmdbForEpg({
        channelId: "x",
        title: "News",
        season: 1,
        episode: 2,
        start: new Date(),
        end: new Date(),
      }),
    ).toBe(true);
  });

  it("allows multi-word show titles", () => {
    expect(
      shouldUseTmdbForEpg({
        channelId: "x",
        title: "We've Lost Dale Earnhardt: 25 Years Later",
        start: new Date(),
        end: new Date(),
      }),
    ).toBe(true);
  });
});
