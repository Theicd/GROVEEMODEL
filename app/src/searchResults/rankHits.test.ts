import { describe, expect, it } from "vitest";
import { filterHits, rankHitsForQuery } from "./rankHits";
import type { UnifiedSearchHit } from "./types";

const wikiColumbus = (): UnifiedSearchHit => ({
  id: "wiki-columbus",
  kind: "web",
  title: "Christopher Columbus",
  titleOriginal: "Christopher Columbus",
  url: "https://en.wikipedia.org/wiki/Christopher_Columbus",
  snippet: "Italian explorer who completed four voyages across the Atlantic Ocean.",
  sourceLabel: "Wikipedia",
  provider: "wikipedia-en",
  score: 44,
  summarizable: true,
});

const ghanaRss = (): UnifiedSearchHit => ({
  id: "gh-1",
  kind: "rss",
  title: "Local football league updates",
  titleOriginal: "Local football league updates",
  url: "https://www.ghanaweb.com/GhanaHomePage/sports/archive/1",
  snippet: "Ghana sports headline unrelated to search.",
  sourceLabel: "GhanaWeb",
  sourceKey: "gh_ghanaweb",
  provider: "grovee-news",
  score: 72,
  summarizable: true,
  meta: { engine: "RSS" },
});

describe("rankHitsForQuery", () => {
  it("ranks Columbus Wikipedia above irrelevant RSS for columbus query", () => {
    const ranked = rankHitsForQuery([ghanaRss(), wikiColumbus()], "columbus", {
      newsQuery: false,
    });
    expect(ranked[0]?.title).toMatch(/Columbus/i);
    expect(ranked[0]?.url).toMatch(/wikipedia/i);
  });

  it("ranks קולומבוס Hebrew query with relevant title first", () => {
    const heWiki: UnifiedSearchHit = {
      ...wikiColumbus(),
      title: "כריסטופר קולומבוס",
      titleOriginal: "כריסטופר קולומבוס",
      provider: "wikipedia-he",
      url: "https://he.wikipedia.org/wiki/כריסטופר_קולומבוס",
    };
    const ranked = rankHitsForQuery([ghanaRss(), heWiki], "קולומבוס", { newsQuery: false });
    expect(ranked[0]?.title).toMatch(/קולומבוס/);
  });

  it("boosts Israeli Hebrew RSS when hebrewUi is on for blended search", () => {
    const ilRss: UnifiedSearchHit = {
      id: "il-1",
      kind: "rss",
      title: "עדכון מבית המשפט",
      titleOriginal: "עדכון מבית המשפט",
      url: "https://www.israelhayom.co.il/news/example",
      snippet: "ידיעה עברית",
      sourceLabel: "ישראל היום",
      sourceKey: "il_israel_hayom",
      provider: "grovee-news",
      score: 40,
      summarizable: true,
      meta: { engine: "RSS" },
    };
    const ranked = rankHitsForQuery([ghanaRss(), ilRss], "מה חדש", {
      newsQuery: false,
      hebrewUi: true,
    });
    expect(ranked[0]?.sourceKey).toBe("il_israel_hayom");
  });
});

describe("filterHits", () => {
  it("filters youtube hits only for youtube tab", () => {
    const yt: UnifiedSearchHit = {
      id: "yt1",
      kind: "youtube",
      title: "Song",
      url: "https://www.youtube.com/watch?v=abc",
      snippet: "",
      sourceLabel: "YouTube",
      provider: "invidious-videos",
      summarizable: false,
    };
    const web: UnifiedSearchHit = {
      id: "w1",
      kind: "web",
      title: "Site",
      url: "https://example.com",
      snippet: "",
      sourceLabel: "example.com",
      provider: "searxng",
      summarizable: true,
    };
    const out = filterHits([yt, web], "youtube");
    expect(out).toHaveLength(1);
    expect(out[0].kind).toBe("youtube");
  });

  it("filters hfmodel hits only for hfmodels tab", () => {
    const hf: UnifiedSearchHit = {
      id: "hf-1",
      kind: "hfmodel",
      title: "Qwen/Qwen2.5-7B-Instruct",
      url: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
      snippet: "Pipeline: text-generation",
      sourceLabel: "Hugging Face",
      provider: "huggingface-models",
      score: 70,
      summarizable: false,
    };
    const web: UnifiedSearchHit = {
      id: "web-1",
      kind: "web",
      title: "Example",
      url: "https://example.com",
      snippet: "x",
      sourceLabel: "Web",
      provider: "searxng",
      score: 40,
      summarizable: true,
    };
    const out = filterHits([hf, web], "hfmodels");
    expect(out).toHaveLength(1);
    expect(out[0]?.kind).toBe("hfmodel");
  });
});
