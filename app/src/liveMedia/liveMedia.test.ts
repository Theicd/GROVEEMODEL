import { describe, expect, it } from "vitest";
import { parseM3U } from "./m3u-parser";
import { parseRadioStations } from "./radio-parser";
import { applyChannelFilters, applyRadioFilters } from "./search";
import { countryMatches, languageMatches } from "./ranking";
import { inferM3UParseDefaults } from "./sourceDefaults";

describe("liveMedia parsers", () => {
  it("parses minimal M3U", () => {
    const m3u = `#EXTM3U
#EXTINF:-1 tvg-logo="http://x/logo.png" group-title="Music;IL",Test Channel
http://example.com/stream.m3u8`;
    const channels = parseM3U(m3u, { source: "test" });
    expect(channels).toHaveLength(1);
    expect(channels[0]?.name).toBe("Test Channel");
    expect(channels[0]?.category).toBe("music");
    expect(channels[0]?.country).toBe("il");
    expect(channels[0]?.stream).toContain("example.com");
  });

  it("infers Israel defaults from country feed URL", () => {
    const defaults = inferM3UParseDefaults({
      id: "iptv-org-il",
      name: "IL",
      type: "iptv",
      url: "https://iptv-org.github.io/iptv/countries/il.m3u",
      enabled: true,
      autoRefresh: true,
      lastSync: 0,
      channelCount: 0,
    });
    expect(defaults.defaultCountry).toBe("il");
  });

  it("matches country via source id when country field empty", () => {
    const ch = {
      id: "1",
      name: "Kan 11",
      logo: "",
      country: "",
      language: "heb",
      category: "news",
      stream: "http://a",
      source: "iptv-org-il",
      type: "tv" as const,
      status: "unknown" as const,
      lastCheck: 0,
      favorite: false,
      addedAt: 0,
    };
    expect(countryMatches(ch, "il")).toBe(true);
    expect(applyChannelFilters([ch], { query: "", country: "il" })).toHaveLength(1);
  });

  it("matches Hebrew language aliases", () => {
    expect(languageMatches("heb", "heb")).toBe(true);
    expect(languageMatches("he", "heb")).toBe(true);
    expect(languageMatches("eng,heb", "heb")).toBe(true);
  });

  it("filters offline channels when onlyWorking", () => {
    const channels = [
      {
        id: "1",
        name: "A",
        logo: "",
        country: "il",
        language: "",
        category: "music",
        stream: "http://a",
        source: "s",
        type: "tv" as const,
        status: "working" as const,
        lastCheck: 0,
        favorite: false,
        addedAt: 0,
      },
      {
        id: "2",
        name: "B",
        logo: "",
        country: "il",
        language: "",
        category: "music",
        stream: "http://b",
        source: "s",
        type: "tv" as const,
        status: "offline" as const,
        lastCheck: 0,
        favorite: false,
        addedAt: 0,
      },
    ];
    expect(applyChannelFilters(channels, { query: "", onlyWorking: true })).toHaveLength(1);
    expect(applyRadioFilters([], { query: "" })).toHaveLength(0);
  });

  it("parses radio JSON row", () => {
    const stations = parseRadioStations([
      {
        stationuuid: "u1",
        name: "Rock FM",
        url: "http://old",
        url_resolved: "http://stream/rock.mp3",
        favicon: "",
        tags: "rock,pop",
        country: "Israel",
        countrycode: "IL",
        language: "he",
        votes: 10,
        codec: "MP3",
        bitrate: 128,
      },
    ]);
    expect(stations[0]?.name).toBe("Rock FM");
    expect(stations[0]?.tags).toContain("rock");
  });
});
