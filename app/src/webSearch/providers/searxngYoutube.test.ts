import { describe, expect, it } from "vitest";

import { webHitToYouTubeMedia } from "./searxngYoutube";

describe("searxngYoutube", () => {
  it("maps SearXNG web row to YouTube media hit", () => {
    const hit = webHitToYouTubeMedia({
      id: "w1",
      title: "שלמה ארצי - שיר לך",
      url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      snippet: "קליפ רשמי",
    });
    expect(hit?.source).toBe("YouTube");
    expect(hit?.playUrl).toContain("embed");
    expect(hit?.thumbnail).toContain("ytimg.com");
  });
});
