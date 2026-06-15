import { describe, expect, it } from "vitest";
import {
  cacheKey,
  clearQueryCache,
  getCachedSearchResult,
  setCachedSearchResult,
  wrapWithQueryCache,
} from "./queryCache";
import type { SearchSourceResult } from "./types";

const mockResult = (provider: SearchSourceResult["provider"]): SearchSourceResult => ({
  provider,
  label: "test",
  ok: true,
  text: "cached data",
  latencyMs: 10,
});

describe("queryCache", () => {
  it("stores and retrieves cached results", () => {
    clearQueryCache();
    const r = mockResult("open-meteo");
    setCachedSearchResult("open-meteo", "weather tel aviv", r);
    const hit = getCachedSearchResult("open-meteo", "weather tel aviv");
    expect(hit?.text).toBe("cached data");
    expect(hit?.latencyMs).toBe(0);
  });

  it("normalizes cache keys", () => {
    expect(cacheKey("news-rss", "  Hello   World ")).toBe("news-rss:hello world");
  });

  it("wrapWithQueryCache calls fetch only once", async () => {
    clearQueryCache();
    let calls = 0;
    const fetch = async () => {
      calls += 1;
      return mockResult("github");
    };
    await wrapWithQueryCache("github", "webgpu", fetch);
    await wrapWithQueryCache("github", "webgpu", fetch);
    expect(calls).toBe(1);
  });

  it("does not cache failed results", () => {
    clearQueryCache();
    setCachedSearchResult("news-rss", "fail", {
      provider: "news-rss",
      label: "news",
      ok: false,
      text: "",
      error: "timeout",
      latencyMs: 1,
    });
    expect(getCachedSearchResult("news-rss", "fail")).toBeNull();
  });
});
