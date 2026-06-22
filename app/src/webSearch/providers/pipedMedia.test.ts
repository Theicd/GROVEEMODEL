import { describe, expect, it } from "vitest";
import { mapPipedChannel, mapPipedPlaylist, mapPipedStream } from "./pipedMedia";

describe("pipedMedia mappers", () => {
  it("maps stream items to YouTube video hits", () => {
    const hit = mapPipedStream({
      type: "stream",
      title: "Demo video",
      url: "/watch?v=dQw4w9WgXcQ",
      thumbnail: "https://example.com/thumb.jpg",
      uploaderName: "Channel",
      duration: 120,
    });
    expect(hit?.youtubeSubType).toBe("video");
    expect(hit?.url).toContain("youtube.com/watch?v=dQw4w9WgXcQ");
  });

  it("maps playlist items", () => {
    const hit = mapPipedPlaylist({
      type: "playlist",
      title: "My playlist",
      playlistId: "PL123",
      videos: 12,
    });
    expect(hit?.youtubeSubType).toBe("playlist");
    expect(hit?.url).toContain("list=PL123");
  });

  it("maps channel items", () => {
    const hit = mapPipedChannel({
      type: "channel",
      uploaderName: "News Channel",
      uploaderUrl: "/channel/UC123",
    });
    expect(hit?.youtubeSubType).toBe("channel");
    expect(hit?.url).toContain("youtube.com");
  });
});
