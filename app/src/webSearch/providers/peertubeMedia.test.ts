import { describe, expect, it } from "vitest";

import {
  mapPeerTubeSearchHit,
  pickPeerTubePlayUrl,
  peerTubeInstanceOrigin,
} from "./peertubeMedia";

describe("peertubeMedia", () => {
  it("picks balanced mp4 rendition", () => {
    const url = pickPeerTubePlayUrl([
      { fileUrl: "https://x/video-1080.mp4", resolution: { label: "1080p" }, width: 1920 },
      { fileUrl: "https://x/video-720.mp4", resolution: { label: "720p" }, width: 1280 },
      { fileUrl: "https://x/video-480.mp4", resolution: { label: "480p" }, width: 854 },
    ]);
    expect(url).toBe("https://x/video-720.mp4");
  });

  it("maps Sepia search row to media hit", () => {
    const hit = mapPeerTubeSearchHit(
      {
        uuid: "abc-123",
        name: "Linux intro",
        url: "https://spectra.video/videos/watch/abc-123",
        thumbnailUrl: "https://spectra.video/thumb.jpg",
        duration: 120,
        account: { displayName: "Tutor" },
      },
      "https://spectra.video/download/abc-123.mp4",
    );
    expect(hit?.source).toBe("PeerTube");
    expect(hit?.playUrl).toContain(".mp4");
    expect(hit?.author).toBe("Tutor");
  });

  it("parses instance origin from watch URL", () => {
    expect(peerTubeInstanceOrigin("https://spectra.video/videos/watch/uuid")).toBe(
      "https://spectra.video",
    );
  });
});
