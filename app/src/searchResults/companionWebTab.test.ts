import { describe, expect, it } from "vitest";
import { filterHits } from "./rankHits";
import type { UnifiedSearchHit } from "./types";
import { webHitSourceLabel } from "./webProviderLabels";

const webHit = (provider: UnifiedSearchHit["provider"], id: string): UnifiedSearchHit => ({
  id,
  kind: "web",
  title: "Title",
  url: "https://example.com/page",
  snippet: "snippet",
  sourceLabel: "x",
  provider,
  summarizable: true,
});

describe("companion web tab", () => {
  it("labels OpenSERP hits with provider and engine", () => {
    expect(webHitSourceLabel("openserp", "https://cinema.co.il/movies", "bing")).toBe(
      "OpenSERP · bing · cinema.co.il",
    );
  });

  it("splits companion vs generic web filters", () => {
    const hits = [
      webHit("openserp", "1"),
      webHit("scavio", "2"),
      webHit("tavily", "3"),
    ];
    expect(filterHits(hits, "companion")).toHaveLength(1);
    expect(filterHits(hits, "companion")[0].provider).toBe("openserp");
    expect(filterHits(hits, "web")).toHaveLength(2);
  });
});
