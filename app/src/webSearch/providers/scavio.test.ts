import { describe, expect, it } from "vitest";
import { mapScavioResultsToWebHits, parseScavioGoogleResponse } from "../../../../vite-plugins/scavioProxy";

describe("scavioProxy", () => {
  it("parses Scavio docs results array", () => {
    const { results, creditsRemaining } = parseScavioGoogleResponse({
      results: [
        {
          title: "Example",
          url: "https://example.com",
          content: "Snippet text here.",
          position: 1,
        },
      ],
      query: "test",
      credits_remaining: 999,
    });
    expect(results).toHaveLength(1);
    expect(creditsRemaining).toBe(999);
    const hits = mapScavioResultsToWebHits(results);
    expect(hits[0].engine).toBe("Scavio Google");
    expect(hits[0].url).toBe("https://example.com");
  });

  it("parses organic_results legacy shape", () => {
    const { results } = parseScavioGoogleResponse({
      organic_results: [
        { title: "Legacy", link: "https://legacy.test", snippet: "old format", position: 2 },
      ],
    });
    expect(results[0].url).toBe("https://legacy.test");
  });
});
