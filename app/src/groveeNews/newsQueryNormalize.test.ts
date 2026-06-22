import { describe, expect, it } from "vitest";
import {
  extractNewsTopicPhrase,
  extractNewsTopicTerms,
  isBroadNewsOverviewQuery,
  isExplicitNewsTopicSearch,
  isSensorNewsQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";

describe("normalizeNewsEngineQuery", () => {
  it("maps Hebrew space news query to English", () => {
    expect(normalizeNewsEngineQuery("חפש חדשות על חלל")).toBe("space");
    expect(normalizeNewsEngineQuery("חדשות על חלל")).toBe("space");
  });

  it("maps earthquake sensor queries to RSS search terms", () => {
    expect(normalizeNewsEngineQuery("רעידות אדמה אחרונות")).toBe("earthquake");
    expect(normalizeNewsEngineQuery("האם היו רעידות מעל 5?")).toBe("earthquake");
  });

  it("maps flood and disaster sensor queries to RSS search terms", () => {
    expect(normalizeNewsEngineQuery("הצפה בטורקיה")).toBe("flood");
    expect(normalizeNewsEngineQuery("הוריקן באירופה")).toBe("disaster");
  });

  it("marks sensor queries without strict Hebrew topic filter", () => {
    expect(isSensorNewsQuery("רעידות אדמה")).toBe(true);
    expect(isSpecificNewsTopicQuery("רעידות אדמה אחרונות")).toBe(false);
  });

  it("maps London from Hebrew", () => {
    expect(normalizeNewsEngineQuery("חפש חדשות על לונדון")).toBe("london");
    expect(normalizeNewsEngineQuery("חדשות על לונדון")).toBe("london");
    expect(extractNewsTopicPhrase("חפש חדשות על לונדון")).toBe("לונדון");
    expect(isSpecificNewsTopicQuery("חפש חדשות על לונדון")).toBe(true);
  });

  it("does not fall back to world news for unknown Hebrew city phrase", () => {
    expect(normalizeNewsEngineQuery("חדשות על זנזיבר")).toBe("");
    expect(isBroadNewsOverviewQuery("")).toBe(true);
    expect(isBroadNewsOverviewQuery("world news")).toBe(true);
  });

  it("maps Iran and Israel topics", () => {
    expect(normalizeNewsEngineQuery("חדשות על איראן")).toBe("iran");
    expect(normalizeNewsEngineQuery("מה קורה בישראל")).toContain("israel");
  });

  it("keeps English tokens from mixed queries", () => {
    expect(normalizeNewsEngineQuery("מה החדשות האחרונות על OpenAI?")).toContain("openai");
  });

  it("falls back to world news for bare Hebrew overview", () => {
    expect(normalizeNewsEngineQuery("מה קורה בעולם?")).toBe("world news");
  });

  it("extractNewsTopicTerms returns tokens", () => {
    expect(extractNewsTopicTerms("חדשות על חלל")).toEqual(["space"]);
    expect(extractNewsTopicTerms("חדשות על לונדון")).toEqual(["london"]);
  });

  it("maps בנושא multi-word topics without marine false positives", () => {
    expect(normalizeNewsEngineQuery("חפש חדשות בנושא טכנולוגיה וסטארטאפים")).toBe("technology startups");
    expect(normalizeNewsEngineQuery("חפש חדשות בנושא סטארטאפים ישראליים")).toBe("israel startups");
    expect(normalizeNewsEngineQuery("חפש חדשות בנושא מלחמות וסכסוכים בעולם")).toBe("war conflict");
  });

  it("explicit בנושא queries are not topics overview", () => {
    const q = "חפש חדשות בנושא מזג אוויר קיצוני בעולם";
    expect(isExplicitNewsTopicSearch(q)).toBe(true);
    expect(normalizeNewsEngineQuery(q)).toBe("extreme weather");
    expect(extractNewsTopicPhrase(q)).toBe("מזג אוויר קיצוני");
  });
});
