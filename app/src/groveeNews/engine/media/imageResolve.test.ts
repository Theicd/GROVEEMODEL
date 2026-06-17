import { describe, expect, it } from "vitest";
import { extractImageFromDescription, resolveWarmArticleImage } from "./imageResolve";

describe("imageResolve warm paths", () => {
  it("uses RSS image without network", () => {
    expect(
      resolveWarmArticleImage({
        articleUrl: "https://news.site/a",
        rssImage: "https://cdn.site/hero.jpg",
      }),
    ).toBe("https://cdn.site/hero.jpg");
  });

  it("extracts embedded img from description", () => {
    const desc = `<p>Lead</p><img src="https://cdn.site/embed.png" />`;
    expect(extractImageFromDescription(desc)).toBe("https://cdn.site/embed.png");
    expect(
      resolveWarmArticleImage({
        articleUrl: "https://news.site/a",
        description: desc,
      }),
    ).toBe("https://cdn.site/embed.png");
  });

  it("ignores stock-library URLs saved by mistake", () => {
    expect(
      resolveWarmArticleImage({
        articleUrl: "https://news.site/a",
        rssImage: "https://cdn.pixabay.com/photo/2024/stock.jpg",
        description: `<img src="https://cdn.site/real.jpg" />`,
      }),
    ).toBe("https://cdn.site/real.jpg");
  });
});
