// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";
import {
  getProviderUsage,
  isProviderEnabled,
  recordProviderUsage,
  resetProviderUsage,
  setProviderEnabled,
} from "./apiProviderUsage";
import { isScavioConfigured, isTavilyConfigured, setScavioApiKey, setTavilyApiKey } from "./apiKeyStore";

describe("apiProviderUsage", () => {
  beforeEach(() => {
    localStorage.clear();
    setTavilyApiKey("");
    setScavioApiKey("");
    resetProviderUsage("tavily");
    resetProviderUsage("scavio");
    setProviderEnabled("tavily", true);
    setProviderEnabled("scavio", true);
  });

  it("defaults providers to enabled", () => {
    expect(isProviderEnabled("tavily")).toBe(true);
    expect(isProviderEnabled("scavio")).toBe(true);
  });

  it("records usage counters and scavio credits", () => {
    const usage = recordProviderUsage("scavio", {
      ok: true,
      hitCount: 5,
      bytesApprox: 2048,
      creditsRemaining: 42,
    });
    expect(usage.requestCount).toBe(1);
    expect(usage.totalHits).toBe(5);
    expect(usage.lastHitCount).toBe(5);
    expect(usage.creditsRemaining).toBe(42);

    recordProviderUsage("scavio", { ok: false, bytesApprox: 100 });
    const stored = getProviderUsage("scavio");
    expect(stored.requestCount).toBe(2);
    expect(stored.successCount).toBe(1);
    expect(stored.creditsRemaining).toBe(42);
  });

  it("disable toggle blocks configured checks", () => {
    setTavilyApiKey("tvly-test-key-12345678");
    expect(isTavilyConfigured()).toBe(true);
    setProviderEnabled("tavily", false);
    expect(isTavilyConfigured()).toBe(false);
    expect(isProviderEnabled("tavily")).toBe(false);
  });

  it("reset clears usage for a provider", () => {
    recordProviderUsage("tavily", { ok: true, hitCount: 3, bytesApprox: 512 });
    resetProviderUsage("tavily");
    expect(getProviderUsage("tavily").requestCount).toBe(0);
  });
});
