import { describe, expect, it } from "vitest";

import {
  archiveDownloadUrl,
  parseArchiveIdentifier,
  pickBestVideoFile,
  promoteArchiveWebHitsToMedia,
} from "./archiveOrgVideo";

describe("archiveOrgVideo", () => {
  it("parses archive.org details URLs", () => {
    expect(parseArchiveIdentifier("https://archive.org/details/TikTok-7449350901090798855")).toBe(
      "TikTok-7449350901090798855",
    );
    expect(parseArchiveIdentifier("https://example.com/foo")).toBeNull();
  });

  it("builds download URLs", () => {
    expect(archiveDownloadUrl("abc", "clip.mp4")).toBe("https://archive.org/download/abc/clip.mp4");
  });

  it("prefers playable mp4 files", () => {
    const file = pickBestVideoFile([
      { name: "meta.xml", format: "Metadata" },
      { name: "clip.ia.mp4", format: "h.264 IA", size: "5000000" },
      { name: "clip.mp4", format: "MPEG4", size: "6000000" },
    ]);
    expect(file?.name).toBe("clip.mp4");
  });

  it("promotes archive web hits with playback to mediaHits", async () => {
    const { webHits, mediaHits } = await promoteArchiveWebHitsToMedia([
      {
        id: "w1",
        title: "Regular page",
        url: "https://example.com/page",
        snippet: "text",
      },
      {
        id: "w2",
        title: "IA video",
        url: "https://archive.org/details/TikTok-7449350901090798855",
        snippet: "chase footage",
      },
    ]);
    expect(webHits).toHaveLength(1);
    expect(webHits[0].url).toContain("example.com");
    if (mediaHits.length) {
      expect(mediaHits[0].mediaType).toBe("video");
      expect(mediaHits[0].playUrl).toMatch(/archive\.org\/download\//);
      expect(mediaHits[0].source).toBe("Internet Archive");
    }
  });
});
