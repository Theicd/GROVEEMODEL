import { describe, expect, it } from "vitest";
import {
  classifySearchIntents,
  extractLocationPhrase,
  extractTimeZonePair,
  isWeatherQuery,
  userRequestsSearch,
  needsWebSearch,
  isCasualConversation,
  isWorldTimeQuery,
  isCountryQuery,
  buildGitHubSearchQuery,
  isMarketPriceQuery,
  isRedditQuery,
  isAviationQuery,
} from "./intents";
import { formatWebContext, summarizeSearchResult } from "./orchestrator";
import type { SearchSourceResult } from "./types";

describe("webSearch intents", () => {
  it("detects weather queries in Hebrew and English", () => {
    expect(isWeatherQuery("מה מזג האוויר בניו יורק")).toBe(true);
    expect(isWeatherQuery("weather in London")).toBe(true);
    expect(isWeatherQuery("hello")).toBe(false);
  });

  it("extracts location from weather question", () => {
    expect(extractLocationPhrase("מה מזג האוויר בניו יורק")).toBeTruthy();
    expect(extractLocationPhrase("weather in Paris")).toBeTruthy();
  });

  it("classifies weather intent", () => {
    const intents = classifySearchIntents("מה מזג האוויר בתל אביב");
    expect(intents).toContain("weather");
  });

  it("classifies earthquake intent", () => {
    expect(classifySearchIntents("רעידות אדמה אחרונות")).toContain("earthquake");
  });

  it("adds wikipedia fallback for general queries", () => {
    expect(classifySearchIntents("מי היה אinstein")).toContain("wikipedia");
  });

  it("detects explicit search request", () => {
    expect(userRequestsSearch("חפש מידע על ברמודה")).toBe(true);
    expect(userRequestsSearch("שלום")).toBe(false);
  });

  it("skips web search for casual greetings", () => {
    expect(isCasualConversation("היי מה שלומך")).toBe(true);
    expect(isCasualConversation("שלום")).toBe(true);
    expect(isCasualConversation("hello")).toBe(true);
    expect(needsWebSearch("היי מה שלומך")).toBe(false);
    expect(needsWebSearch("שלום")).toBe(false);
  });

  it("enables web search for factual and live queries", () => {
    expect(needsWebSearch("מה מזג האוויר בתל אביב")).toBe(true);
    expect(needsWebSearch("מה גובה הגלים בחיפה")).toBe(true);
    expect(needsWebSearch("מי היה אinstein")).toBe(true);
    expect(needsWebSearch("חפש מידע על ברמודה")).toBe(true);
  });

  it("does not add wikipedia for pure weather intent", () => {
    expect(classifySearchIntents("מה מזג האוויר בתל אביב")).toContain("weather");
  });

  it("classifies world time and country intents", () => {
    expect(classifySearchIntents("מה השעה בטוקיו")).toContain("worldtime");
    expect(classifySearchIntents("מה הבירה של גרמניה")).toEqual(["country"]);
    expect(classifySearchIntents("האם היום חג בגרמניה")).toContain("holiday");
    expect(classifySearchIntents("מי ראש הממשלה של ישראל")).toContain("government");
    expect(classifySearchIntents("שער USD ל ILS")).toContain("currency");
    expect(classifySearchIntents("מה המטבע של ברזיל")).toEqual(["country"]);
    expect(classifySearchIntents("כמה BRL אני קונה ב1 דולר")).toContain("currency");
    expect(classifySearchIntents("כמה ק\"מ בין ירושלים לחיפה")).toContain("distance");
    expect(classifySearchIntents("מצא בית חולים ליד מגדל אייפel")).toContain("places");
    expect(classifySearchIntents("אילו תחנות רכבת יש ליד שדה התעופה הית'רו")).toContain("places");
    expect(classifySearchIntents("מה הכותרת הראשית באתר BBC עכשיו")).toContain("news");
  });

  it("detects new query types for needsWebSearch", () => {
    expect(needsWebSearch("מה השעה בלונדון")).toBe(true);
    expect(needsWebSearch("מה הבירה של צרפת")).toBe(true);
    expect(isWorldTimeQuery("what time in Tokyo")).toBe(true);
    expect(isCountryQuery("population of Israel")).toBe(true);
  });

  it("builds github query from latin tokens", () => {
    expect(buildGitHubSearchQuery("open source llm chat")).toContain("llm");
  });

  it("builds github query from hebrew tech hints", () => {
    const q = buildGitHubSearchQuery("פרויקט github למצלמות אבטחה");
    expect(q.length).toBeGreaterThan(5);
  });

  it("classifies timezone offset between two places", () => {
    expect(classifySearchIntents("כמה שעות הפרש יש בין ישראל לאוסטרליה")).toContain("worldtime");
    const pair = extractTimeZonePair("כמה שעות הפרש יש בין ישראל לאוסטרליה");
    expect(pair?.[0]).toMatch(/ישראל/i);
    expect(pair?.[1]).toMatch(/אוסטרליה/i);
  });

  it("skips wikipedia for market and reddit queries", () => {
    expect(classifySearchIntents("מה מחיר מניית NVIDIA")).toEqual(["market"]);
    expect(classifySearchIntents("מה קורה ב-r/worldnews")).toEqual(["reddit"]);
    expect(isMarketPriceQuery("מחיר זהב היום")).toBe(true);
    expect(isRedditQuery("reddit trending")).toBe(true);
  });

  it("routes explicit web search to searx and wikipedia", () => {
    const intents = classifySearchIntents("חפש מידע על פירמידות");
    expect(intents).toContain("wikipedia");
    expect(intents).toContain("searx");
  });

  it("classifies hacker news and arxiv intents", () => {
    expect(classifySearchIntents("מה חם ב-hacker news")).toContain("hackernews");
    expect(classifySearchIntents("מאמרים על transformers arxiv")).toContain("arxiv");
  });

  it("detects aviation follow-up without repeating aircraft keyword", () => {
    expect(isAviationQuery("כמה מהם צבאיים")).toBe(true);
    expect(classifySearchIntents("כמה מהם צבאיים")).toContain("aviation");
  });

  it("builds trending github query from hebrew", () => {
    expect(buildGitHubSearchQuery("פרויקטים פופולריים בגיטהב השבוע")).toContain("stars:");
  });
});

describe("formatWebContext", () => {
  it("formats ok sources with grounding header", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "open-meteo",
        label: "מזג אוויר",
        ok: true,
        text: "טמפרatura: 22°C",
        url: "https://open-meteo.com",
        latencyMs: 100,
      },
    ];
    const ctx = formatWebContext(sources);
    expect(ctx).toContain("WEB SEARCH RESULTS");
    expect(ctx).toContain("22°C");
    expect(ctx).toContain("open-meteo.com");
  });

  it("returns empty when all failed", () => {
    expect(
      formatWebContext([
        {
          provider: "github",
          label: "GitHub",
          ok: false,
          text: "",
          error: "fail",
          latencyMs: 1,
        },
      ]),
    ).toBe("");
  });

  it("summarizes hebrew status", () => {
    const s = summarizeSearchResult(
      [
        {
          provider: "open-meteo",
          label: "מזג אוויר",
          ok: true,
          text: "x",
          latencyMs: 1,
        },
      ],
      ["weather"],
    );
    expect(s).toContain("1 מקורות");
  });
});
