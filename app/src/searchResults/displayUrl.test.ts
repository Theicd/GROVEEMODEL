import { describe, expect, it } from "vitest";
import { displayBreadcrumb, googleTranslatePageUrl } from "./displayUrl";

describe("displayBreadcrumb", () => {
  it("formats hostname and path segments with ellipsis", () => {
    const url = "https://www.thehindu.com/news/national/telangana/five-held-for-posing-as-task-force-officers";
    const crumb = displayBreadcrumb(url, 48);
    expect(crumb).toContain("thehindu.com");
    expect(crumb).toContain("›");
    expect(crumb.length).toBeLessThanOrEqual(49);
  });

  it("truncates long paths", () => {
    const url = "https://github.com/cirosantilli/china-dictatorship/tree/main/very/long/path/here";
    const crumb = displayBreadcrumb(url, 40);
    expect(crumb.endsWith("…") || crumb.includes("› …")).toBe(true);
  });
});

describe("googleTranslatePageUrl", () => {
  it("builds translate.google.com link", () => {
    const out = googleTranslatePageUrl("https://example.com/page", "he");
    expect(out).toContain("translate.google.com");
    expect(out).toContain("tl=he");
    expect(out).toContain(encodeURIComponent("https://example.com/page"));
  });
});
