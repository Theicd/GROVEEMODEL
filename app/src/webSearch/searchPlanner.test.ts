import { describe, expect, it } from "vitest";
import {
  parseSearchPlanJson,
  regexPlanForQuery,
  shouldUseSearchPlanner,
} from "./searchPlanner";

describe("searchPlanner", () => {
  it("parses valid JSON plan — intents from rules not JSON", () => {
    const raw = `{"intents":["weather"],"queries":["weather Tel Aviv"],"answerShape":"short_fact"}`;
    const plan = parseSearchPlanJson(raw, "מה מזג האוויר בתל אביב");
    expect(plan?.intents).toContain("weather");
    expect(plan?.queries[0]).toMatch(/Tel Aviv/i);
    expect(plan?.useWebFallback).toBe(false);
  });

  it("planner useWebFallback only when no rule intents", () => {
    const raw = `{"queries":["robotics trends"],"answerShape":"overview","useWebFallback":true}`;
    const plan = parseSearchPlanJson(raw, "מה קורה בעולם הרובוטיקה?");
    expect(plan?.intents).toEqual([]);
    expect(plan?.useWebFallback).toBe(true);
  });

  it("regex plan for air traffic paraphrase", () => {
    const plan = regexPlanForQuery("מה העומס בשמי ישראל?");
    expect(plan?.intents).toContain("aviation");
  });

  it("regex plan for clear weather skips planner", () => {
    expect(regexPlanForQuery("מה מזג האוויר בתל אביב")?.intents).toContain("weather");
    expect(shouldUseSearchPlanner("מה מזג האוויר בתל אביב")).toBe(false);
  });

  it("regex plan handles world events without Gemma", () => {
    expect(regexPlanForQuery("מה קורה עכשיו בעולם?")?.intents).toContain("news");
    expect(shouldUseSearchPlanner("מה קורה עכשיו בעולם?")).toBe(false);
  });

  it("regex plan for URL link", () => {
    const plan = regexPlanForQuery("https://github.com/Theicd/GROVEEMODEL");
    expect(plan?.intents).toEqual(["link"]);
  });

  it("topical world query uses multi-source enrichment in regex plan", () => {
    const plan = regexPlanForQuery("מה קורה בעולם הגיימינג?");
    expect(plan?.useWebFallback).toBe(true);
    expect(plan?.blendNewsWithWeb).toBe(true);
    expect(plan?.intents).toContain("hackernews");
    expect(plan?.intents).toContain("github");
    expect(plan?.intents).toContain("news");
  });

  it("regex plan for robotics includes arxiv and RSS blend", () => {
    const plan = regexPlanForQuery("מה קורה בעולם הרובוטיקה?");
    expect(plan?.useWebFallback).toBe(true);
    expect(plan?.intents).toContain("arxiv");
    expect(plan?.intents).toContain("news");
  });

  it("explicit search blends RSS and web", () => {
    const plan = regexPlanForQuery("חפש חדשות על כלכלה");
    expect(plan?.blendNewsWithWeb).toBe(true);
    expect(plan?.useWebFallback).toBe(true);
    expect(plan?.intents).toContain("news");
  });
});
