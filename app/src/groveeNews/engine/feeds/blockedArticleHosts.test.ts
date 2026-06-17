import { describe, expect, it } from "vitest";
import { isBlockedArticleHost } from "./blockedArticleHosts";

describe("blockedArticleHosts", () => {
  it("blocks food sites that return 403 to scrapers", () => {
    expect(isBlockedArticleHost("https://www.thekitchn.com/some-recipe")).toBe(true);
    expect(isBlockedArticleHost("https://www.bonappetit.com/story")).toBe(true);
    expect(isBlockedArticleHost("https://www.eater.com/2024/1/1")).toBe(true);
    expect(isBlockedArticleHost("https://www.bbc.com/news")).toBe(false);
  });
});
