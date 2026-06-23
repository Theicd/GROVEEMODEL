import { describe, expect, it } from "vitest";
import {
  buildCuratedFavoritesFile,
  curatedSnapshotToChannel,
  injectCuratedChannels,
  mergeCuratedFavoritesIntoPrefs,
} from "./curatedFavorites";
import { emptyUserPrefs } from "./userPrefs";
import type { Channel, RadioStation } from "./types";

const sampleChannel = (id: string, name: string): Channel => ({
  id,
  name,
  logo: "",
  country: "il",
  language: "heb",
  category: "news",
  stream: `http://example/${id}`,
  source: "test",
  type: "tv",
  status: "working",
  lastCheck: 0,
  favorite: true,
  addedAt: 0,
});

const sampleRadio = (id: string, name: string): RadioStation => ({
  id,
  name,
  favicon: "",
  tags: ["news"],
  country: "Israel",
  countrycode: "IL",
  language: "hebrew",
  stream: `http://radio/${id}`,
  type: "radio",
  favorite: true,
  addedAt: 0,
});

describe("curatedFavorites", () => {
  it("merges repo channel and radio ids into local prefs", () => {
    const prefs = emptyUserPrefs();
    prefs.favoriteChannelIds = ["local-tv"];
    const merged = mergeCuratedFavoritesIntoPrefs(prefs, {
      version: 1,
      updatedAt: 1,
      channels: [{ id: "repo-tv", name: "Repo TV", country: "il", language: "heb", category: "news", stream: "x", source: "s", type: "tv" }],
      radio: [{ id: "repo-rd", name: "Repo RD", country: "IL", countrycode: "IL", language: "he", stream: "y", tags: [] }],
    });
    expect(merged.changed).toBe(true);
    expect(merged.prefs.favoriteChannelIds.sort()).toEqual(["local-tv", "repo-tv"].sort());
    expect(merged.prefs.favoriteRadioIds).toEqual(["repo-rd"]);
  });

  it("builds git snapshot from resolved favorites only", () => {
    const prefs = emptyUserPrefs();
    prefs.favoriteChannelIds = ["a", "missing"];
    prefs.favoriteRadioIds = ["r1"];
    const file = buildCuratedFavoritesFile(
      [sampleChannel("a", "Alpha"), sampleChannel("b", "Beta")],
      [sampleRadio("r1", "Radio One")],
      prefs,
    );
    expect(file.version).toBe(1);
    expect(file.channels.map((c) => c.id)).toEqual(["a"]);
    expect(file.radio.map((r) => r.id)).toEqual(["r1"]);
    expect(file.channels[0]?.name).toBe("Alpha");
  });

  it("injects curated snapshots when catalog is still empty", () => {
    const snap = {
      id: "repo-tv",
      name: "Repo TV",
      country: "il",
      language: "heb",
      category: "news",
      stream: "https://example/stream.m3u8",
      source: "iptv-org-all",
      type: "tv" as const,
    };
    const injected = injectCuratedChannels([], {
      version: 1,
      updatedAt: 1,
      channels: [snap],
      radio: [],
    });
    expect(injected).toHaveLength(1);
    expect(injected[0]?.id).toBe("repo-tv");
    expect(injected[0]?.favorite).toBe(true);
    expect(curatedSnapshotToChannel(snap).stream).toBe(snap.stream);
  });
});
