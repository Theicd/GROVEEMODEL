import { describe, expect, it } from "vitest";
import { classifySearchIntents } from "../webSearch/intents";
import { resolveLiveDataHandoff } from "../webSearch/liveDataHandoff";
import { mergeSourcesToHits, buildUnifiedSearchPayload } from "../searchResults/mergeSearchHits";
import { filterHits } from "../searchResults/rankHits";
import {
  isSensorNewsQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import {
  isTopHitRelevant,
  rankRssHeadlinesForQuery,
  rssItemToSearchArticle,
} from "./engine/search/relevance";
import type { RssItem } from "./engine/types";
import type { SearchSourceResult } from "../webSearch/types";

const eqRss: RssItem = {
  id: "eq-rss-1",
  title: "Breaking news: earthquake emergency disaster response",
  description: "Rescue teams rush after major seismic tremor.",
  source: "BBC",
  sourceKey: "bbc",
  category: "world",
  link: "https://news.example.com/eq1",
  image: "",
  published: new Date().toISOString(),
  publishedTs: Date.now(),
  guid: "eq-rss-1",
};

const floodRss: RssItem = {
  id: "fl-rss-1",
  title: "Extreme weather floods coastal cities",
  description: "Storm systems bring heavy rain and inundation worldwide.",
  source: "Reuters",
  sourceKey: "reuters",
  category: "science",
  link: "https://news.example.com/fl1",
  image: "",
  published: new Date().toISOString(),
  publishedTs: Date.now(),
  guid: "fl-rss-1",
};

describe("Sensor + RSS QA — רעידות אדמה / הצפה", () => {
  const earthquakeQueries = [
    "רעידות אדמה אחרונות",
    "האם היו רעידות מעל 5 ב-24 שעות",
    "earthquake Japan magnitude 6",
  ];

  const floodQueries = ["הצפה בטורקיה", "flood disaster Europe", "שיטפון אחרון"];

  it.each(earthquakeQueries)("earthquake query %s → news intent + RSS terms", (q) => {
    const intents = classifySearchIntents(q);
    expect(intents).toContain("earthquake");
    expect(intents).toContain("news");
    expect(normalizeNewsEngineQuery(q)).toBe("earthquake");
    const handoff = resolveLiveDataHandoff(q);
    expect(handoff.providers).toContain("grovee-news");
    expect(handoff.providers).toContain("usgs-earthquake");
  });

  it.each(floodQueries)("flood/disaster query %s → news + gdacs", (q) => {
    const intents = classifySearchIntents(q);
    expect(intents).toContain("disaster");
    expect(intents).toContain("news");
    expect(normalizeNewsEngineQuery(q)).toMatch(/^flood$|^disaster$/);
    expect(resolveLiveDataHandoff(q).providers).toContain("grovee-news");
  });

  it("sensor queries are not over-filtered as specific Hebrew-only topics", () => {
    for (const q of [...earthquakeQueries, ...floodQueries]) {
      expect(isSensorNewsQuery(q)).toBe(true);
      expect(isSpecificNewsTopicQuery(q)).toBe(false);
    }
  });

  it("RSS corpus returns earthquake disaster headlines", () => {
    const ranked = rankRssHeadlinesForQuery([eqRss, floodRss], "earthquake", new Set(), 10);
    expect(ranked.length).toBeGreaterThan(0);
    const article = rssItemToSearchArticle(eqRss);
    expect(isTopHitRelevant(article, "earthquake")).toBe(true);
  });

  it("RSS corpus returns flood headlines", () => {
    const ranked = rankRssHeadlinesForQuery([eqRss, floodRss], "flood", new Set(), 10);
    expect(ranked.length).toBeGreaterThan(0);
    expect(isTopHitRelevant(rssItemToSearchArticle(floodRss), "flood")).toBe(true);
  });

  it("mergeSourcesToHits keeps RSS cards alongside USGS for earthquake search", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "usgs-earthquake",
        label: "USGS",
        ok: true,
        text: '- M5.2 · Japan · 2024-06-19 12:00:00 UTC\n  https://earthquake.usgs.gov/x',
        latencyMs: 100,
      },
      {
        provider: "grovee-news",
        label: "חדשות",
        ok: true,
        text: "ANSWER (headline): [BBC] Breaking earthquake",
        latencyMs: 200,
        newsCards: [
          {
            id: "n1",
            title: "Breaking earthquake emergency in Japan",
            titleOriginal: "Breaking earthquake emergency in Japan",
            source: "BBC",
            sourceKey: "bbc",
            url: "https://bbc.com/eq",
            image: "",
            score: 80,
            publishedTs: Date.now(),
          },
        ],
      },
    ];
    const hits = mergeSourcesToHits(sources, "רעידות אדמה אחרונות");
    expect(hits.some((h) => h.kind === "earthquake")).toBe(true);
    expect(hits.some((h) => h.kind === "rss")).toBe(true);

    const payload = buildUnifiedSearchPayload("רעידות אדמה אחרונות", sources);
    expect(payload.facets.rss).toBeGreaterThan(0);
    expect(payload.facets.earthquakes).toBeGreaterThan(0);
    expect(payload.preferEventsFilter).toBe(false);
    expect(filterHits(payload.hits, "rss").length).toBeGreaterThan(0);
  });
});
