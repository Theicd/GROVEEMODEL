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
  buildHuggingFaceSearchQuery,
  isCommodityPriceQuery,
  isRedditQuery,
  isAviationQuery,
} from "./intents";
import { extractCurrencyPair, sanitizeSearchQuery } from "./queryExtract";
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

  it("does not add wikipedia without explicit search request", () => {
    expect(classifySearchIntents("מי היה אinstein")).toEqual([]);
    expect(classifySearchIntents("חפש מידע על איינשטיין")).toContain("wikipedia");
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

  it("auto search only for live or explicit lookup — not static facts", () => {
    expect(needsWebSearch("מה מזג האוויר בתל אביב")).toBe(true);
    expect(needsWebSearch("מה גובה הגלים בחיפה")).toBe(true);
    expect(needsWebSearch("מה השעה בלונדון")).toBe(true);
    expect(needsWebSearch("כמה מטוסים נמצאים כרגע מעל ישראל?")).toBe(true);
    expect(needsWebSearch("חפש מידע על ברמודה")).toBe(true);
    expect(needsWebSearch("מי היה אinstein")).toBe(false);
    expect(needsWebSearch("מה בירת יפן?")).toBe(false);
    expect(needsWebSearch("מה המטבע של ברזיל?")).toBe(false);
    expect(needsWebSearch('כמה ק"מ בין ירושלים לחיפה?')).toBe(false);
    expect(needsWebSearch("מי ראש הממשלה של בריטניה?")).toBe(true);
    expect(needsWebSearch("מה מחיר חבית נפט Brent?")).toBe(true);
  });

  it("does not add wikipedia for pure weather intent", () => {
    expect(classifySearchIntents("מה מזג האוויר בתל אביב")).toEqual(["weather"]);
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

  it("detects live query types for needsWebSearch", () => {
    expect(needsWebSearch("מה השעה בלונדון")).toBe(true);
    expect(needsWebSearch("מה הבירה של צרפת")).toBe(false);
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
    expect(classifySearchIntents("מה קורה ב-r/worldnews")).toEqual([]);
    expect(isCommodityPriceQuery("מחיר זהב היום")).toBe(true);
    expect(isRedditQuery("reddit trending")).toBe(true);
  });

  it("detects aviation follow-up without repeating aircraft keyword", () => {
    expect(isAviationQuery("כמה מהם צבאיים")).toBe(true);
    expect(classifySearchIntents("כמה מהם צבאיים")).toContain("aviation");
  });

  it("does not add wikipedia for github trending query", () => {
    expect(classifySearchIntents("פרויקטים פופולריים בגיטהב השבוע")).toEqual(["github"]);
  });

  it("sanitizes greeting from time query location", () => {
    const cleaned = sanitizeSearchQuery("בוקר טוב מר גרובי, איזה יום ומה השעה בישראל");
    expect(cleaned).not.toMatch(/בוקר\s+טוב/i);
    expect(cleaned).toMatch(/ישrael|ישראל/i);
  });

  it("builds trending github query from hebrew", () => {
    expect(buildGitHubSearchQuery("פרויקטים פופולריים בגיטהב השבוע")).toContain("stars:");
  });

  it("routes currency conversion without wikipedia", () => {
    expect(classifySearchIntents("כמה יורו שווים 1000 שקלים?")).toEqual(["currency"]);
    expect(classifySearchIntents("כמה שקלים שווים 100 דולר?")).toEqual(["currency"]);
    expect(classifySearchIntents("כמה BRL מקבלים עבור 1 דולר?")).toEqual(["currency"]);
  });

  it("extracts currency pair with amount from natural hebrew", () => {
    expect(extractCurrencyPair("כמה יורו שווים 1000 שקלים?")).toEqual({
      from: "ILS",
      to: "EUR",
      amount: 1000,
    });
    expect(extractCurrencyPair("כמה BRL מקבלים עבור 1 דולר?")).toEqual({
      from: "USD",
      to: "BRL",
      amount: 1,
    });
  });

  it("routes population question to country provider, not wikipedia", () => {
    expect(classifySearchIntents("כמה תושבים יש בקנדה?")).toEqual(["country"]);
  });

  it("does not misroute generic 'מהם' questions to aviation", () => {
    expect(isAviationQuery("מהם 10 הנושאים המסוקרים ביותר בעולם כרגע?")).toBe(false);
    expect(classifySearchIntents("מהם 10 הנושאים המסוקרים ביותר בעולם כרגע?")).not.toContain("aviation");
    expect(classifySearchIntents("מהם מודלי התמונה הפופולריים ביותר השבוע?")).toEqual(["huggingface"]);
  });

  it("extracts location from wind speed question", () => {
    expect(extractLocationPhrase("מה מהירות הרוח בפריז?")).toMatch(/פריז|paris/i);
  });

  it("extracts Madrid from Hebrew forecast question", () => {
    expect(extractLocationPhrase("מהי תחזית מזג האוויר במדריד למחר?")).toMatch(/מדריד|madrid/i);
  });

  it("classifies OCR search to huggingface and github", () => {
    expect(classifySearchIntents("חפש מודלים ל-OCR")).toEqual(
      expect.arrayContaining(["huggingface", "github"]),
    );
  });

  it("routes AI tech news to hackernews", () => {
    expect(needsWebSearch("מה קורה כרגע בתחום הבינה המלאכותית?")).toBe(true);
    expect(classifySearchIntents("מה קורה כרגע בתחום הבינה המלאכותית?")).toContain("hackernews");
  });

  it("routes flight status at airport", () => {
    expect(needsWebSearch("מה מצב הטיסות בנמל התעופה JFK?")).toBe(true);
    expect(classifySearchIntents("Nearest train station Eiffel Tower")).toContain("places");
  });

  it("routes ships in Suez and satellite catalog", () => {
    expect(classifySearchIntents("אילו ספינות בתעלת סואץ")).toContain("ships");
    expect(classifySearchIntents("כמה לוויינים פעילים יש כיום?")).toContain("satellite");
    expect(classifySearchIntents("מהו השיגור הקרוב של SpaceX?")).toContain("spacex");
    expect(classifySearchIntents("מהו הפוסט הפופולרי ביותר ב-Hacker News כרגע?")).toContain("hackernews");
  });

  it("does not add wikipedia for crypto price query", () => {
    expect(classifySearchIntents("מה מחיר הביטקוין עכשיו?")).toEqual(["crypto"]);
  });

  it("routes commodity prices and current government leaders to live search", () => {
    expect(isCommodityPriceQuery("מה מחיר חבית נפט Brent?")).toBe(true);
    expect(classifySearchIntents("מה מחיר חבית נפט Brent?")).toContain("commodity");
    expect(classifySearchIntents("מי ראש הממשלה של בריטניה?")).toContain("government");
    expect(classifySearchIntents("מה מצב מדד S&P 500?")).toEqual(["market"]);
    expect(needsWebSearch("מה מצב מדד S&P 500?")).toBe(true);
  });

  it("builds huggingface image model query", () => {
    expect(buildHuggingFaceSearchQuery("מהם מודלי התמונה הפופולריים ביותר השבוע?")).toBe("stable-diffusion");
  });

  it("does not treat earthquake duration as world time", () => {
    expect(isWorldTimeQuery("האם הייתה רעידת אדמה ביפן ב-48 השעות האחרונות?")).toBe(false);
    const intents = classifySearchIntents("האם הייתה רעידת אדמה ביפן ב-48 השעות האחרונות?");
    expect(intents).toContain("earthquake");
    expect(intents).not.toContain("worldtime");
    expect(classifySearchIntents("האם הייתה רעידת אדמה בישראל השבוע?")).toContain("earthquake");
  });

  it("does not classify Hacker News as BBC news", () => {
    const intents = classifySearchIntents("מהו הפוסט הפופולרי ביותר ב-Hacker News כרגע?");
    expect(intents).toContain("hackernews");
    expect(intents).not.toContain("news");
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
    expect(ctx).toContain("SEARCH BRIEF");
    expect(ctx).toContain("22°C");
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
