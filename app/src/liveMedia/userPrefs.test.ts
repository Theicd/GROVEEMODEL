import { describe, expect, it, vi } from "vitest";
import {
  emptyUserPrefs,
  migrateLegacyFavoritesIntoPrefs,
  releaseBlacklistedFavorites,
} from "./userPrefs";
import type { Channel, RadioStation } from "./types";

vi.mock("./indexeddb", () => ({
  dbPutUserPrefs: vi.fn(),
}));

describe("userPrefs favorites recovery", () => {
  it("migrates legacy favorite flags from channel rows into prefs", async () => {
    const channels: Channel[] = [
      {
        id: "a",
        name: "A",
        logo: "",
        country: "il",
        language: "heb",
        category: "news",
        stream: "http://a",
        source: "s",
        type: "tv",
        status: "working",
        lastCheck: 0,
        favorite: true,
        addedAt: 0,
      },
      {
        id: "b",
        name: "B",
        logo: "",
        country: "us",
        language: "eng",
        category: "movies",
        stream: "http://b",
        source: "s",
        type: "tv",
        status: "unknown",
        lastCheck: 0,
        favorite: true,
        addedAt: 0,
      },
    ];
    const prefs = emptyUserPrefs();
    prefs.favoriteChannelIds = ["c"];
    const next = await migrateLegacyFavoritesIntoPrefs(channels, [], prefs);
    expect(next.favoriteChannelIds.sort()).toEqual(["a", "b", "c"].sort());
  });

  it("removes favorites from blacklist on repair", () => {
    const prefs = {
      ...emptyUserPrefs(),
      favoriteChannelIds: ["star-tv"],
      blacklistChannelIds: ["star-tv", "junk-tv"],
    };
    const next = releaseBlacklistedFavorites(prefs);
    expect(next.blacklistChannelIds).toEqual(["junk-tv"]);
  });
});
