import { describe, expect, it } from "vitest";

import { parseYouTubeVideoId, youtubeEmbedUrl, youtubeThumbnail } from "./youtubeUrls";

describe("youtubeUrls", () => {
  it("parses watch URLs", () => {
    expect(parseYouTubeVideoId("https://www.youtube.com/watch?v=dQw4w9WgXcQ")).toBe("dQw4w9WgXcQ");
    expect(parseYouTubeVideoId("https://youtu.be/dQw4w9WgXcQ")).toBe("dQw4w9WgXcQ");
  });

  it("builds thumbnail and embed URLs", () => {
    expect(youtubeThumbnail("abc123")).toContain("abc123");
    expect(youtubeEmbedUrl("abc123")).toContain("embed/abc123");
  });
});
