import { describe, expect, it, vi, afterEach } from "vitest";
import { hasDirectCors, needsProxy, isStaticWebHost } from "./proxyFetch";

describe("proxyFetch web deployment", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("detects GitHub Pages as static web host", () => {
    vi.stubGlobal("window", {
      location: { hostname: "theicd.github.io", port: "", origin: "https://theicd.github.io" },
    });
    expect(isStaticWebHost()).toBe(true);
  });

  it("open-meteo allows direct CORS", () => {
    expect(hasDirectCors("https://api.open-meteo.com/v1/forecast?latitude=32")).toBe(true);
  });

  it("BBC RSS needs proxy on static host", () => {
    vi.stubGlobal("window", {
      location: { hostname: "theicd.github.io", port: "", origin: "https://theicd.github.io" },
    });
    expect(needsProxy("https://feeds.bbci.co.uk/news/rss.xml")).toBe(true);
  });
});
