import { describe, expect, it } from "vitest";
import { IL_HE_FEEDS } from "./catalog/regions/il-he";
import { ALL_CATALOG_FEEDS } from "./catalog";
import { WORLD_LANG_FEEDS } from "./catalog/world-lang-feeds";
import { WORLD_PERIPHERY_FEEDS } from "./catalog/world-periphery-feeds";
import { resolveActiveFeeds } from "./feedRegistry";
import { needsTranslation } from "../topics/buildWorldFeedBundle";

describe("world lang feeds", () => {
  it("includes multilingual sources", () => {
    const langs = new Set(WORLD_LANG_FEEDS.map((f) => f.lang));
    expect(langs.has("he")).toBe(true);
    expect(langs.has("fr")).toBe(true);
    expect(langs.has("de")).toBe(true);
    expect(langs.has("ru")).toBe(true);
    expect(langs.has("ja")).toBe(true);
    expect(langs.has("zh")).toBe(true);
    expect(langs.has("es")).toBe(true);
    expect(langs.has("ar")).toBe(true);
    expect(langs.has("ko")).toBe(true);
    expect(langs.has("it")).toBe(true);
    expect(langs.has("pt")).toBe(true);
    expect(langs.has("uk")).toBe(true);
  });

  it("covers underserved regions via periphery pack", () => {
    const regions = new Set(WORLD_PERIPHERY_FEEDS.map((f) => f.region));
    expect(regions.has("TR")).toBe(true);
    expect(regions.has("IN")).toBe(true);
    expect(regions.has("AF")).toBe(true);
    expect(regions.has("NG")).toBe(true);
    expect(regions.has("ZA")).toBe(true);
    expect(regions.has("ID")).toBe(true);
    expect(regions.has("PH")).toBe(true);
    expect(regions.has("AR")).toBe(true);
    expect(regions.has("AU")).toBe(true);
    expect(WORLD_PERIPHERY_FEEDS.length).toBeGreaterThanOrEqual(25);
  });
});

describe("resolveActiveFeeds", () => {
  it("returns global catalog for all users", () => {
    const feeds = resolveActiveFeeds({ locale: "fr-FR", uiLanguage: "fr", pollTier: "core" });
    expect(feeds.length).toBeGreaterThan(50);
    expect(feeds.some((f) => f.key === "fr_lemonde")).toBe(true);
    expect(feeds.some((f) => f.key === "il_makor_rishon")).toBe(true);
  });

  it("merges world lang and IL feeds into catalog", () => {
    expect(ALL_CATALOG_FEEDS.some((f) => f.key === "ja_nhk")).toBe(true);
    expect(ALL_CATALOG_FEEDS.filter((f) => f.key.startsWith("il_")).length).toBeGreaterThanOrEqual(IL_HE_FEEDS.length);
  });
});

describe("needsTranslation", () => {
  it("skips translation when feed matches ui language", () => {
    expect(needsTranslation("fr", "fr")).toBe(false);
    expect(needsTranslation("he", "he")).toBe(false);
    expect(needsTranslation("en", "en")).toBe(false);
  });

  it("translates when feed lang differs from ui", () => {
    expect(needsTranslation("en", "fr")).toBe(true);
    expect(needsTranslation("he", "en")).toBe(true);
    expect(needsTranslation("ja", "ru")).toBe(true);
  });
});
