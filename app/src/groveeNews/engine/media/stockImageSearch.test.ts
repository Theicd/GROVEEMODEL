import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  buildImageSearchQuery,
  clearStockImageCacheForTests,
  detectStockProvider,
  listConfiguredStockProviders,
  searchStockImage,
} from "./stockImageSearch";

vi.mock("../fetch/remoteFetch", () => ({
  fetchRemoteText: vi.fn(),
}));

import { fetchRemoteText } from "../fetch/remoteFetch";

const fetchMock = vi.mocked(fetchRemoteText);

describe("buildImageSearchQuery export", () => {
  it("drops stop words and keeps topic terms", () => {
    expect(buildImageSearchQuery("Champions League football final preview")).toMatch(/champions/i);
  });
});

describe("detectStockProvider", () => {
  it("recognizes known stock hosts", () => {
    expect(detectStockProvider("https://images.pexels.com/photos/1/test.jpeg")).toBe("pexels");
    expect(detectStockProvider("https://www.bbc.com/news")).toBeNull();
  });
});

describe("searchStockImage", () => {
  beforeEach(() => {
    clearStockImageCacheForTests();
    fetchMock.mockReset();
    vi.stubEnv("VITE_PIXABAY_API_KEY", "test-key");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("picks best-scoring candidate from multiple Pixabay hits", async () => {
    fetchMock.mockResolvedValueOnce(
      JSON.stringify({
        hits: [
          {
            tags: "abstract background texture",
            largeImageURL: "https://cdn.pixabay.com/photo/bad.jpg",
            imageWidth: 1920,
            imageHeight: 1080,
          },
          {
            tags: "mars rover nasa space planet",
            largeImageURL: "https://cdn.pixabay.com/photo/good.jpg",
            imageWidth: 1920,
            imageHeight: 1080,
          },
        ],
      }),
    );

    const hit = await searchStockImage("NASA Mars rover", "space");
    expect(hit?.provider).toBe("pixabay");
    expect(hit?.url).toContain("good.jpg");
  });

  it("lists always-on providers without optional keys", () => {
    vi.unstubAllEnvs();
    const providers = listConfiguredStockProviders();
    expect(providers).toContain("openverse");
    expect(providers).toContain("pexels");
  });
});
