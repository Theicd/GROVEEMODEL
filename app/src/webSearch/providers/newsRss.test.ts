import { describe, expect, it, vi, beforeEach } from "vitest";
import { fetchNewsSearch, parseRssTitles, selectNewsFeedKeys } from "./newsRss";
import { extractNewsSite, isWorldHeadlineQuery, isGeneralNewsDigestQuery } from "../queryExtract";

vi.mock("../fetchJson", () => ({
  fetchText: vi.fn(),
}));

import { fetchText } from "../fetchJson";

const mockFetch = vi.mocked(fetchText);

const rss = (titles: string[]) =>
  `<rss><channel>${titles.map((t) => `<item><title>${t}</title></item>`).join("")}</channel></rss>`;

describe("news RSS", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("detects general news digest as multi-feed", () => {
    expect(isGeneralNewsDigestQuery("ספר לי מה חדש היום בחדשות")).toBe(true);
    expect(extractNewsSite("ספר לי מה חדש היום בחדשות")).toBeNull();
    expect(selectNewsFeedKeys("ספר לי מה חדש היום בחדשות").length).toBeGreaterThanOrEqual(3);
  });

  it("detects B01 as world headline query", () => {
    expect(isWorldHeadlineQuery("מה הכותרת הראשית בעולם כרגע?")).toBe(true);
    expect(extractNewsSite("מה הכותרת הראשית בעולם כרגע?")).toBeNull();
  });

  it("keeps BBC-only for explicit BBC query", () => {
    expect(extractNewsSite("מה הכותרת הראשית באתר BBC עכשיו")).toBe("bbc");
  });

  it("parses RSS titles", () => {
    expect(parseRssTitles(rss(["A", "B"]), 5)).toEqual(["A", "B"]);
  });

  it("aggregates multiple international feeds for world query", async () => {
    mockFetch.mockImplementation(async (url: string) => {
      if (url.includes("bbci")) return rss(["BBC headline"]);
      if (url.includes("cnn")) return rss(["CNN headline"]);
      if (url.includes("reuters")) return rss(["Reuters headline"]);
      if (url.includes("guardian")) return rss(["Guardian headline"]);
      return rss([]);
    });

    const result = await fetchNewsSearch("מה הכותרת הראשית בעולם כרגע?");
    expect(result.ok).toBe(true);
    expect(result.text).toContain("ANSWER (headline): [BBC] BBC headline");
    expect(result.text).toMatch(/\[CNN\]/);
    expect(result.text).toMatch(/\[Reuters\]/);
    expect(result.text).toMatch(/\[Guardian\]/);
    expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(4);
  });

  it("uses single feed for BBC-specific query", async () => {
    mockFetch.mockResolvedValue(rss(["Only BBC"]));

    const result = await fetchNewsSearch("מה הכותרת הראשית באתר BBC עכשיו");
    expect(result.ok).toBe(true);
    expect(result.text).toContain("[BBC]");
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});
