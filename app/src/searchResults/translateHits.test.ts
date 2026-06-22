import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UnifiedSearchHit } from "./types";
import { translateSearchHits } from "./translateHits";

const translateTextsMock = vi.fn();

vi.mock("../groveeNews/engine/translate/googleTranslate", () => ({
  translateTexts: (...args: unknown[]) => translateTextsMock(...args),
}));

describe("translateSearchHits", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const enHit = (id: string): UnifiedSearchHit => ({
    id,
    kind: "web",
    title: "Breaking news headline",
    titleOriginal: "Breaking news headline",
    snippet: "Short English snippet",
    snippetOriginal: "Short English snippet",
    url: `https://example.com/${id}`,
    sourceLabel: "Example",
    provider: "wikipedia-en",
    summarizable: false,
  });

  it("skips translation when UI language matches content", async () => {
    const hits: UnifiedSearchHit[] = [
      {
        id: "1",
        kind: "rss",
        title: "כותרת חדשות",
        titleOriginal: "כותרת חדשות",
        snippet: "תקציר קצר",
        snippetOriginal: "תקציר קצר",
        url: "https://example.com/1",
        sourceLabel: "ynet",
        provider: "grovee-news",
        summarizable: true,
      },
    ];
    const out = await translateSearchHits(hits, "he");
    expect(translateTextsMock).not.toHaveBeenCalled();
    expect(out[0].title).toBe("כותרת חדשות");
  });

  it("translates titles and snippets to Hebrew UI", async () => {
    translateTextsMock
      .mockResolvedValueOnce({ texts: ["כותרת חדשות"], provider: "cache" })
      .mockResolvedValueOnce({ texts: ["תקציר קצר"], provider: "cache" });

    const out = await translateSearchHits([enHit("1")], "he");
    expect(translateTextsMock).toHaveBeenCalledTimes(2);
    expect(out[0].title).toBe("כותרת חדשות");
    expect(out[0].snippet).toBe("תקציר קצר");
    expect(out[0].titleOriginal).toBe("Breaking news headline");
  });

  it("returns originals when translate API fails", async () => {
    translateTextsMock.mockRejectedValue(new Error("503 Service Unavailable"));
    const hits = [enHit("1")];
    const out = await translateSearchHits(hits, "he");
    expect(out[0].title).toBe("Breaking news headline");
    expect(out[0].snippet).toBe("Short English snippet");
  });
});
