import { describe, expect, it } from "vitest";
import { pixabayVideoThumbnail } from "./pixabayMedia";

describe("pixabayVideoThumbnail", () => {
  it("uses API thumbnail from video rendition", () => {
    const url = pixabayVideoThumbnail({
      id: 42,
      pageURL: "https://pixabay.com/videos/butterfly-42/",
      user: "adege",
      tags: "butterfly",
      duration: 29,
      picture_id: "593165292",
      videos: {
        medium: {
          url: "https://cdn.pixabay.com/video/medium.mp4",
          thumbnail: "https://cdn.pixabay.com/video/thumb-medium.jpg",
        },
      },
    });
    expect(url).toBe("https://cdn.pixabay.com/video/thumb-medium.jpg");
  });

  it("falls back to Vimeo CDN poster when API thumbnail missing", () => {
    const url = pixabayVideoThumbnail({
      id: 99,
      pageURL: "https://pixabay.com/videos/x-99/",
      user: "u",
      tags: "x",
      duration: 10,
      picture_id: "529927645",
      videos: {
        medium: { url: "https://cdn.pixabay.com/video/medium.mp4" },
      },
    });
    expect(url).toBe("https://i.vimeocdn.com/video/529927645_295x166.jpg");
  });
});
