import { describe, expect, it } from "vitest";
import { channelToSearchHit } from "./adapters";
import { applyPrefsToHit } from "./channelDisplay";
import { filterTunerFavorites, groupHitsByUserCategory, sortHitsByCategoryOrder } from "./liveMediaFilters";
import type { Channel } from "./types";
import { emptyUserPrefs } from "./userPrefs";

function ch(partial: Partial<Channel> & Pick<Channel, "id" | "name">): Channel {
  return {
    type: "tv",
    stream: "https://example.com/stream.m3u8",
    source: "iptv-org-il",
    category: "general",
    country: "il",
    language: "heb",
    favorite: true,
    ...partial,
  };
}

describe("liveMediaFilters", () => {
  it("filters tuner favorites by category and language", () => {
    const news = ch({ id: "n1", name: "Now 14", category: "news" });
    const sports = ch({ id: "s1", name: "Sport 5", category: "sports", language: "eng" });
    const channels = [news, sports];
    const prefs = {
      ...emptyUserPrefs(),
      tunerEnabledCategories: ["news"],
      viewLanguages: ["heb"],
    };
    const hits = channels.map((c) => applyPrefsToHit(channelToSearchHit(c), c, prefs));
    const filtered = filterTunerFavorites(hits, channels, prefs, "il");
    expect(filtered.map((h) => h.id)).toEqual([`livetv-${news.id}`]);
  });

  it("groups favorites by user category", () => {
    const news = ch({ id: "n1", name: "Now 14", category: "news" });
    const movies = ch({ id: "m1", name: "Movie Channel", category: "movies" });
    const channels = [news, movies];
    const prefs = emptyUserPrefs();
    const hits = channels.map((c) => applyPrefsToHit(channelToSearchHit(c), c, prefs));
    const grouped = groupHitsByUserCategory(hits, channels, prefs);
    expect(grouped.get("news")?.length).toBe(1);
    expect(grouped.get("movies")?.length).toBe(1);
  });

  it("applies custom image URL to display hit", () => {
    const c = ch({ id: "i1", name: "Broken Logo", logo: "" });
    const prefs = {
      ...emptyUserPrefs(),
      channelOverrides: {
        [c.id]: { imageUrl: "https://example.com/logo.png" },
      },
    };
    const hit = applyPrefsToHit(channelToSearchHit(c), c, prefs);
    expect(hit.imageUrl).toBe("https://example.com/logo.png");
  });

  it("applies custom stream URL to display hit", () => {
    const c = ch({ id: "f1", name: "FIFA+", stream: "https://catalog.example.com/fifa.m3u8" });
    const custom = "https://mirror.example.com/fifa-us.m3u8";
    const prefs = {
      ...emptyUserPrefs(),
      channelOverrides: {
        [c.id]: { streamUrl: custom },
      },
    };
    const hit = applyPrefsToHit(channelToSearchHit(c), c, prefs);
    expect(hit.mediaPlayUrl).toBe(custom);
    expect(hit.url).toBe(custom);
  });

  it("sorts tuner favorites by user category order", () => {
    const sports = ch({ id: "s1", name: "Sport 5", category: "sports" });
    const movies = ch({ id: "m1", name: "Movie Channel", category: "movies" });
    const channels = [sports, movies];
    const prefs = {
      ...emptyUserPrefs(),
      tunerCategoryOrder: ["movies", "sports"],
    };
    const hits = channels.map((c) => applyPrefsToHit(channelToSearchHit(c), c, prefs));
    const sorted = sortHitsByCategoryOrder(hits, channels, prefs);
    expect(sorted.map((h) => h.id)).toEqual([`livetv-${movies.id}`, `livetv-${sports.id}`]);
  });
});
