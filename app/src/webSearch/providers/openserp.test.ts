import { describe, expect, it, vi, beforeEach } from "vitest";

const fetchMock = vi.fn();

vi.mock("../../plugins/search-companion/companionSettings", () => ({
  getSearchCompanionServiceUrl: () => "http://127.0.0.1:7000",
  resolveSearchCompanionFetchBase: () => "http://127.0.0.1:5180/api/openserp",
  setSearchCompanionUrl: vi.fn(),
  usesDevOpenSerpProxy: () => true,
}));

vi.mock("../../plugins/search-companion/health", () => ({
  isSearchCompanionReachable: () => false,
}));

vi.mock("./openserpWebMedia", () => ({
  promoteCompanionWebHitsToMedia: async (hits: unknown[]) => ({ webHits: hits, mediaHits: [] }),
}));

import { fetchOpenSerpSearch } from "./openserp";

describe("fetchOpenSerpSearch", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  it("fetches web and image megasearch in parallel", async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.includes("/mega/image")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            results: [
              {
                id: "img1",
                type: "image",
                title: "Sunset",
                image: { url: "https://cdn.example/s.jpg", thumbnail: "https://cdn.example/t.jpg" },
                source: { page_url: "https://example.com/s", domain: "example.com" },
                engine: "bing",
              },
            ],
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          results: [
            {
              title: "WebGPU",
              url: "https://example.com/webgpu",
              snippet: "GPU in browser",
              engine: "bing",
            },
          ],
          meta: { took_ms: 1200 },
        }),
      });
    });

    const out = await fetchOpenSerpSearch("webgpu", { limit: 5 });
    expect(out.ok).toBe(true);
    expect(out.webHits?.[0].title).toBe("WebGPU");
    expect(out.mediaHits?.[0].mediaType).toBe("image");
    expect(out.mediaHits?.[0].source).toContain("OpenSERP");
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/mega/search"))).toBe(true);
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("/mega/image"))).toBe(true);
  });

  it("returns error when no web or media results", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ results: [], meta: { engines_failed: ["google"] } }),
    });
    const out = await fetchOpenSerpSearch("xyzempty");
    expect(out.ok).toBe(false);
    expect(out.error).toContain("google");
  });
});
