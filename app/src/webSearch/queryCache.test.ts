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
  it("stores and retrieves cached results for static providers", () => {
    clearQueryCache();
    const r = mockResult("github");
    setCachedSearchResult("github", "webgpu repos", r);
    const hit = getCachedSearchResult("github", "webgpu repos");
    expect(hit?.text).toBe("cached data");
    expect(hit?.latencyMs).toBe(0);
  });

  it("does not persist live provider results across turns", () => {
    clearQueryCache();
    setCachedSearchResult("usgs-earthquake", "רעידות אדמה", mockResult("usgs-earthquake"));
    expect(getCachedSearchResult("usgs-earthquake", "רעידות אדמה")).toBeNull();
  });

  it("normalizes cache keys", () => {
    expect(cacheKey("grovee-news", "  Hello   World ")).toBe("grovee-news:hello world");
  });

  it("wrapWithQueryCache calls fetch on every live provider request", async () => {
    clearQueryCache();
    let calls = 0;
    const fetch = async () => {
      calls += 1;
      return mockResult("usgs-earthquake");
    };
    await wrapWithQueryCache("usgs-earthquake", "earthquakes", fetch);
    await wrapWithQueryCache("usgs-earthquake", "earthquakes", fetch);
    expect(calls).toBe(2);
  });

  it("wrapWithQueryCache dedupes static providers within same session", async () => {
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
    setCachedSearchResult("grovee-news", "fail", {
      provider: "grovee-news",
      label: "news",
      ok: false,
      text: "",
      error: "timeout",
      latencyMs: 1,
    });
    expect(getCachedSearchResult("grovee-news", "fail")).toBeNull();
  });
});
