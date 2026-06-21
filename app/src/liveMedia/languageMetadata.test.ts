import { describe, expect, it } from "vitest";
import { matchesDefaultBlacklistChannel } from "./defaultBlacklist";
import { enrichChannel, inferChannelLanguages, languageDisplayLabel } from "./languageMetadata";
import type { Channel } from "./types";

const baseChannel = (overrides: Partial<Channel>): Channel => ({
  id: "1",
  name: "Test",
  logo: "",
  country: "",
  language: "",
  category: "general",
  stream: "http://x",
  source: "test",
  type: "tv",
  status: "unknown",
  lastCheck: 0,
  favorite: false,
  addedAt: 0,
  ...overrides,
});

describe("languageMetadata", () => {
  it("infers Hindi from channel name", () => {
    const langs = inferChannelLanguages(
      baseChannel({ name: "B4U Bhojpuri (1080p)", category: "movies" }),
    );
    expect(langs).toContain("hin");
  });

  it("infers Hebrew from Israel country feed", () => {
    const langs = inferChannelLanguages(
      baseChannel({ name: "Kan 11", country: "il", source: "iptv-org-il" }),
    );
    expect(langs).toContain("heb");
  });

  it("enrichChannel sets languages array", () => {
    const c = enrichChannel(baseChannel({ name: "BBC News", language: "eng" }));
    expect(c.languages).toContain("eng");
    expect(languageDisplayLabel("eng", true)).toBe("אנגלית");
  });
});

describe("defaultBlacklist", () => {
  it("blacklists religious category", () => {
    expect(matchesDefaultBlacklistChannel(baseChannel({ category: "religious", name: "Faith TV" }))).toBe(true);
  });

  it("blacklists bhojpuri movie clutter", () => {
    expect(matchesDefaultBlacklistChannel(baseChannel({ name: "B4U Bhojpuri HD", category: "movies" }))).toBe(true);
  });

  it("keeps generic english movie channel", () => {
    expect(matchesDefaultBlacklistChannel(baseChannel({ name: "MovieSphere (1080p)", category: "movies", language: "eng" }))).toBe(
      false,
    );
  });
});
