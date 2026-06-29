import { describe, expect, it } from "vitest";
import {
  defaultBroadcastLanguageForChannel,
  isIsraeliChannel,
  mapCatalogCategoryToUser,
  normalizeChannelImageUrl,
  normalizeChannelStreamUrl,
  resolveBroadcastLanguage,
  resolveChannelImageUrl,
  resolveChannelStreamUrl,
  resolveUserCategory,
} from "./channelUserTaxonomy";
import type { Channel } from "./types";

function ch(partial: Partial<Channel> & Pick<Channel, "id" | "name">): Channel {
  return {
    type: "tv",
    stream: "https://example.com/stream.m3u8",
    source: "iptv-org-all",
    category: "general",
    country: "",
    language: "",
    ...partial,
  };
}

describe("channelUserTaxonomy", () => {
  it("maps Now 14 to news category", () => {
    const c = ch({ id: "x1", name: "Now 14 (1080p)", country: "il", language: "heb" });
    expect(mapCatalogCategoryToUser(c)).toBe("news");
    expect(resolveUserCategory(c.id, c, {})).toBe("news");
  });

  it("resolves Hebrew for Israeli kan channel", () => {
    const c = ch({
      id: "x2",
      name: "Kan 11 (1080p)",
      tvgId: "Kan11.il@SD",
      country: "il",
      source: "iptv-org-il",
    });
    expect(resolveBroadcastLanguage(c)).toBe("heb");
    expect(isIsraeliChannel(c)).toBe(true);
  });

  it("defaults international channels to English", () => {
    const c = ch({
      id: "x4",
      name: "AMC Absolute Reality (1080p)",
      tvgId: "AbsoluteRealitybyWETV.us@SD",
      country: "",
    });
    expect(defaultBroadcastLanguageForChannel(c)).toBe("eng");
    expect(resolveBroadcastLanguage(c)).toBe("eng");
    expect(isIsraeliChannel(c)).toBe(false);
  });

  it("ignores mistaken Hebrew override on US channel", () => {
    const c = ch({ id: "x5", name: "FOX Sports", tvgId: "FoxSports.us@SD" });
    expect(resolveBroadcastLanguage(c, { broadcastLanguage: "heb" })).toBe("eng");
  });

  it("maps music and documentary catalog categories", () => {
    expect(mapCatalogCategoryToUser(ch({ id: "m1", name: "MTV Hits", category: "music" }))).toBe("music");
    expect(mapCatalogCategoryToUser(ch({ id: "d1", name: "Discovery", category: "documentary" }))).toBe(
      "documentary",
    );
    expect(mapCatalogCategoryToUser(ch({ id: "g1", name: "Misc", category: "general" }))).toBe("general");
  });

  it("honors user override for category and language", () => {
    const c = ch({ id: "x3", name: "Some Channel", category: "movies" });
    const cat = resolveUserCategory(c.id, c, { [c.id]: { category: "kids", broadcastLanguage: "rus" } });
    expect(cat).toBe("kids");
    expect(resolveBroadcastLanguage(c, { category: "kids", broadcastLanguage: "rus" })).toBe("rus");
  });

  it("normalizes and resolves custom channel image URL", () => {
    expect(normalizeChannelImageUrl("ftp://bad/logo.png")).toBeUndefined();
    expect(normalizeChannelImageUrl("https://cdn.example.com/logo.png")).toBe("https://cdn.example.com/logo.png");
    const c = ch({ id: "img1", name: "No Logo", logo: "" });
    const custom = "https://example.com/kan11.png";
    expect(resolveChannelImageUrl(c.id, c, { [c.id]: { imageUrl: custom } })).toBe(custom);
  });

  it("normalizes and resolves custom channel stream URL", () => {
    expect(normalizeChannelStreamUrl("ftp://bad/stream.m3u8")).toBeUndefined();
    const custom = "https://cdn.example.com/live/stream.m3u8";
    expect(normalizeChannelStreamUrl(custom)).toBe(custom);
    const c = ch({ id: "s1", name: "FIFA+", stream: "https://catalog.example.com/fifa.m3u8" });
    expect(resolveChannelStreamUrl(c.id, c, { [c.id]: { streamUrl: custom } })).toBe(custom);
    expect(resolveChannelStreamUrl(c.id, c, {})).toBe("https://catalog.example.com/fifa.m3u8");
  });
});
