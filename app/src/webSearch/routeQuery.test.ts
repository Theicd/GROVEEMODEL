import { describe, expect, it } from "vitest";
import { routeQuery, shouldAllowWebFallback, primaryTierForIntents } from "./routeQuery";
import { classifySearchIntents } from "./intents";

describe("routeQuery", () => {
  it("routes weather to structured tier without web fallback", () => {
    const route = routeQuery("מה מזג האוויר בתל אביב");
    expect(route.intents).toContain("weather");
    expect(route.tier).toBe("structured");
    expect(route.useWebFallback).toBe(false);
  });

  it("routes URL paste to link intent", () => {
    const route = routeQuery("https://github.com/Theicd/GROVEEMODEL");
    expect(route.intents).toEqual(["link"]);
    expect(route.tier).toBe("structured");
    expect(route.useWebFallback).toBe(false);
  });

  it("routes robotics topic to multi-source enrichment + optional web", () => {
    const route = routeQuery("מה קורה בעולם הרובוטיקה?");
    expect(route.intents).toContain("hackernews");
    expect(route.intents).toContain("github");
    expect(route.intents).toContain("arxiv");
    expect(route.intents).toContain("news");
    expect(route.useWebFallback).toBe(true);
    expect(route.blendNewsWithWeb).toBe(true);
  });

  it("allows SearXNG alongside news RSS for overview blend", () => {
    expect(
      shouldAllowWebFallback(1, { blendNewsWithWeb: true, useWebFallback: true }, "מה קורה בעולם?"),
    ).toBe(true);
    expect(shouldAllowWebFallback(2, {}, "מה מזג האוויר")).toBe(false);
  });

  it("allows web fallback when plan explicitly requests it", () => {
    expect(shouldAllowWebFallback(2, { useWebFallback: true }, "מה מזג האוויר")).toBe(true);
  });

  it("classifies news tier", () => {
    expect(primaryTierForIntents(["news"])).toBe("news");
    expect(primaryTierForIntents(["news", "weather"])).toBe("structured");
  });

  it("routes Israel headline query to news RSS only", () => {
    const route = routeQuery("מה הכותרות הראשיות היום בישראל");
    expect(route.intents).toEqual(["news"]);
    expect(route.useWebFallback).toBe(false);
  });

  it("cross-source storm query expands intents", () => {
    const intents = classifySearchIntents("האם יש סופה פעילה באירופה?");
    expect(intents).toContain("disaster");
    expect(intents).toContain("weather");
  });
});
