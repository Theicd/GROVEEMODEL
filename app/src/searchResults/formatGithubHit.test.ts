import { describe, expect, it } from "vitest";
import { formatGithubTitleLine } from "./formatGithubHit";
import { translateSearchHits } from "./translateHits";
import type { UnifiedSearchHit } from "./types";

describe("formatGithubTitleLine", () => {
  it("splits repo and description", () => {
    expect(formatGithubTitleLine("seerr-team/seerr: Open media request manager")).toEqual({
      repo: "seerr-team/seerr",
      description: "Open media request manager",
    });
  });
});

describe("translateSearchHits", () => {
  it("does not duplicate title into snippet for GitHub hits", async () => {
    const hits: UnifiedSearchHit[] = [
      {
        id: "g1",
        kind: "github",
        title: "foo/bar: A cool project",
        url: "https://github.com/foo/bar",
        snippet: "",
        sourceLabel: "GitHub",
        provider: "github",
        summarizable: false,
      },
    ];

    const out = await translateSearchHits(hits, "he");
    expect(out[0].snippet).toBe("");
    expect(out[0].title).toBeTruthy();
  });
});
