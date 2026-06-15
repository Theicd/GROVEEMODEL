import { describe, expect, it } from "vitest";
import {
  extractUrlsFromText,
  hasUrlInQuery,
  isGitHubRepoUrlInQuery,
  parseUrl,
  primaryUrlFromQuery,
} from "./urlExtract";

describe("urlExtract", () => {
  it("extracts GitHub repo URL from text", () => {
    const url = "https://github.com/Theicd/GROVEEMODEL";
    expect(hasUrlInQuery(url)).toBe(true);
    expect(primaryUrlFromQuery(url)).toBe(url);
    expect(isGitHubRepoUrlInQuery(url)).toBe(true);
  });

  it("parses GitHub repo parts", () => {
    const parsed = parseUrl("https://github.com/Theicd/GROVEEMODEL");
    expect(parsed.kind).toBe("github-repo");
    if (parsed.kind === "github-repo") {
      expect(parsed.owner).toBe("Theicd");
      expect(parsed.repo).toBe("GROVEEMODEL");
    }
  });

  it("extracts multiple URLs", () => {
    const urls = extractUrlsFromText("ראה https://a.com/x וגם https://b.com/y");
    expect(urls).toHaveLength(2);
  });

  it("parses arXiv link", () => {
    const parsed = parseUrl("https://arxiv.org/abs/2301.00001");
    expect(parsed.kind).toBe("arxiv");
  });
});
