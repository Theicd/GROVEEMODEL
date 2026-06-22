import { describe, expect, it } from "vitest";
import { promoteCompanionWebHitsToMedia } from "./openserpWebMedia";

describe("promoteCompanionWebHitsToMedia", () => {
  it("promotes Vimeo watch URLs to playable video hits", async () => {
    const { webHits, mediaHits } = await promoteCompanionWebHitsToMedia([
      {
        id: "1",
        title: "Demo reel",
        url: "https://vimeo.com/123456789",
        snippet: "short film",
      },
    ]);
    expect(webHits).toHaveLength(0);
    expect(mediaHits).toHaveLength(1);
    expect(mediaHits[0].playUrl).toContain("player.vimeo.com/video/123456789");
  });

  it("promotes PeerTube watch URLs", async () => {
    const { mediaHits } = await promoteCompanionWebHitsToMedia([
      {
        id: "2",
        title: "PT clip",
        url: "https://peertube.cpy.re/videos/watch/abc-def-123",
        snippet: "",
      },
    ]);
    expect(mediaHits[0]?.playUrl).toContain("/videos/embed/abc-def-123");
    expect(mediaHits[0]?.source).toBe("PeerTube");
  });

  it("keeps normal web URLs in webHits", async () => {
    const { webHits, mediaHits } = await promoteCompanionWebHitsToMedia([
      {
        id: "3",
        title: "Article",
        url: "https://example.com/post",
        snippet: "text",
      },
    ]);
    expect(webHits).toHaveLength(1);
    expect(mediaHits).toHaveLength(0);
  });
});
