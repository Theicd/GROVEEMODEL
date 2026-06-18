import { describe, expect, it } from "vitest";

import { invidiousEmbedUrl, mapInvidiousHit, pickInvidiousThumbnail } from "./invidiousMedia";

describe("invidiousMedia", () => {
  it("builds embed URL for in-app iframe player", () => {
    expect(invidiousEmbedUrl("https://inv.example", "dQw4w9WgXcQ")).toBe(
      "https://inv.example/embed/dQw4w9WgXcQ",
    );
  });

  it("maps API row to media hit with embed play URL", () => {
    const hit = mapInvidiousHit("https://inv.example", {
      type: "video",
      videoId: "abc123",
      title: "Demo clip",
      author: "Channel",
      lengthSeconds: 90,
      videoThumbnails: [{ quality: "medium", url: "https://img/m.jpg", width: 320 }],
    });
    expect(hit?.mediaType).toBe("video");
    expect(hit?.playUrl).toContain("/embed/abc123");
    expect(hit?.source).toBe("YouTube");
    expect(pickInvidiousThumbnail(hit ? [{ url: hit.thumbnail, width: 320 }] : [])).toBe(
      hit?.thumbnail,
    );
  });
});
