import { describe, expect, it } from "vitest";
import { buildFetchAttempts, hasLocalFetchProxy } from "./remoteFetch";

describe("remoteFetch routing", () => {
  it("uses local proxy only in dev on localhost", () => {
    expect(hasLocalFetchProxy(true, "127.0.0.1")).toBe(true);
    expect(hasLocalFetchProxy(true, "localhost")).toBe(true);
    expect(hasLocalFetchProxy(false, "theicd.github.io")).toBe(false);
    expect(hasLocalFetchProxy(true, "theicd.github.io")).toBe(true);
  });

  it("puts configured proxy first on static hosting", () => {
    const attempts = buildFetchAttempts("https://example.com/feed.xml", {
      dev: false,
      hostname: "theicd.github.io",
      proxyUrl: "https://proxy.example/worker",
    });
    expect(attempts[0]).toContain("proxy.example");
    expect(attempts.some((u) => u.startsWith("/api/fetch"))).toBe(false);
  });

  it("uses only local proxy in dev (no CORS relays)", () => {
    const attempts = buildFetchAttempts("https://example.com/feed.xml", {
      dev: true,
      hostname: "127.0.0.1",
      proxyUrl: "",
    });
    expect(attempts).toHaveLength(1);
    expect(attempts[0]).toContain("/api/fetch");
    expect(attempts.some((u) => u.includes("allorigins"))).toBe(false);
  });

  it("skips public relays on GitHub Pages without configured proxy", () => {
    const attempts = buildFetchAttempts("https://example.com/feed.xml", {
      dev: false,
      hostname: "theicd.github.io",
      proxyUrl: "",
    });
    expect(attempts.some((u) => u.includes("allorigins"))).toBe(false);
    expect(attempts.some((u) => u.includes("cors.lol"))).toBe(false);
  });
});
