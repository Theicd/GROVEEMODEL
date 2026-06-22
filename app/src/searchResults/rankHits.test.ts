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
  it("filters youtube hits for youtube tab and includes them in video tab", () => {
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
    const videoOut = filterHits([yt, web], "video");
    expect(videoOut).toHaveLength(1);
    expect(videoOut[0].kind).toBe("youtube");
  });

  it("excludes live sensor hits from all tab", () => {
    const ship: UnifiedSearchHit = {
      id: "s1",
      kind: "ship",
      title: "SHIP",
      url: "https://example.com",
      snippet: "",
      sourceLabel: "AIS",
      provider: "ais-ships",
      summarizable: false,
    };
    const web: UnifiedSearchHit = {
      id: "w1",
      kind: "web",
      title: "Wiki",
      url: "https://example.com",
      snippet: "",
      sourceLabel: "Web",
      provider: "openserp",
      summarizable: true,
    };
    expect(filterHits([ship, web], "all")).toEqual([web]);
    expect(filterHits([ship, web], "ships")).toEqual([ship]);
  });

  it("routes weather and sea conditions to events; marine infra to ships", () => {
    const weather: UnifiedSearchHit = {
      id: "w",
      kind: "weather",
      title: "Tel Aviv 25°C",
      url: "https://open-meteo.com/",
      snippet: "",
      sourceLabel: "Open-Meteo",
      provider: "open-meteo",
      summarizable: false,
    };
    const sea: UnifiedSearchHit = {
      id: "m1",
      kind: "marine",
      title: "Waves Haifa",
      url: "https://open-meteo.com/",
      snippet: "",
      sourceLabel: "Marine",
      provider: "open-meteo-marine",
      summarizable: false,
    };
    const buoy: UnifiedSearchHit = {
      id: "m2",
      kind: "marine",
      title: "Buoy",
      url: "https://osm.org",
      snippet: "",
      sourceLabel: "OSM",
      provider: "osm-overpass-marine",
      summarizable: false,
    };
    const hits = [weather, sea, buoy];
    expect(filterHits(hits, "events")).toEqual([weather, sea]);
    expect(filterHits(hits, "ships")).toEqual([buoy]);
  });

  it("excludes image and video hits from all tab", () => {
    const image: UnifiedSearchHit = {
      id: "i1",
      kind: "image",
      title: "Photo",
      url: "https://example.com/img.jpg",
      snippet: "",
      sourceLabel: "Pixabay",
      provider: "pixabay-images",
      summarizable: false,
    };
    const video: UnifiedSearchHit = {
      id: "v1",
      kind: "video",
      title: "Clip",
      url: "https://example.com/v.mp4",
      snippet: "",
      sourceLabel: "Pixabay",
      provider: "pixabay-videos",
      summarizable: false,
    };
    const web: UnifiedSearchHit = {
      id: "w1",
      kind: "web",
      title: "Article",
      url: "https://example.com",
      snippet: "",
      sourceLabel: "Web",
      provider: "tavily",
      summarizable: true,
    };
    expect(filterHits([image, video, web], "all")).toEqual([web]);
    expect(filterHits([image, video, web], "images")).toEqual([image]);
    expect(filterHits([image, video, web], "video")).toEqual([video]);
  });

  it("includes place and route hits in all tab", () => {
    const place: UnifiedSearchHit = {
      id: "p1",
      kind: "place",
      title: "Berlin",
      url: "https://www.openstreetmap.org/",
      snippet: "52.52, 13.40",
      sourceLabel: "OpenStreetMap",
      provider: "nominatim-places",
      summarizable: false,
    };
    const route: UnifiedSearchHit = {
      id: "r1",
      kind: "route",
      title: "Tel Aviv → Jerusalem",
      url: "https://www.openstreetmap.org/",
      snippet: "45 min",
      sourceLabel: "OpenStreetMap",
      provider: "nominatim-places",
      summarizable: false,
    };
    const web: UnifiedSearchHit = {
      id: "w1",
      kind: "web",
      title: "Article",
      url: "https://example.com",
      snippet: "",
      sourceLabel: "Web",
      provider: "tavily",
      summarizable: true,
    };
    expect(filterHits([place, route, web], "all")).toEqual([place, route, web]);
    expect(filterHits([place, route, web], "places")).toEqual([place, route]);
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
