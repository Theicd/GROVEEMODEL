import { describe, expect, it } from "vitest";
import { mapTavilyResultsToWebHits } from "../../../../vite-plugins/tavilyProxy";

describe("tavilyProxy", () => {
  it("maps Tavily API rows to web SERP hits", () => {
    const hits = mapTavilyResultsToWebHits([
      {
        title: "Example Site",
        url: "https://example.com/page",
        content: "Some summary text about the page.",
        score: 0.91,
      },
    ]);
    expect(hits).toHaveLength(1);
    expect(hits[0].title).toBe("Example Site");
    expect(hits[0].engine).toBe("Tavily");
    expect(hits[0].snippet).toContain("summary");
  });
});
