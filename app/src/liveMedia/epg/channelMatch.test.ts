import { describe, expect, it } from "vitest";
import { findBestChannelMatch, scoreChannelMatch } from "./channelMatch";
import type { EpgChannelRef } from "./types";

const channels: EpgChannelRef[] = [
  { id: "romance-movies", name: "Romance Movies", sourceKey: "mjh-all" },
  { id: "gravitas-1", name: "Gravitas Movies", sourceKey: "plex" },
  { id: "gravitas-movies-us", name: "Gravitas Movies", sourceKey: "plex" },
  { id: "bet-tyler", name: "BET x Tyler Perry Comedy", sourceKey: "plex" },
  { id: "mjh-10-cops", name: "COPS", sourceKey: "mjh-all" },
  { id: "pluto-et", name: "ET", sourceKey: "pluto" },
];

describe("channelMatch", () => {
  it("does not match Gravitas Movies to Romance Movies via generic 'movies'", () => {
    const ch = findBestChannelMatch(channels, "Gravitas Movies (1080p)", "GravitasMovies.us@SD");
    expect(ch?.name).toBe("Gravitas Movies");
    expect(scoreChannelMatch(channels[0], "Gravitas Movies (1080p)", "GravitasMovies.us@SD")).toBe(0);
  });

  it("does not match Reshet 13 to ET", () => {
    expect(findBestChannelMatch(channels, "Reshet 13 (720p)")).toBeNull();
    expect(scoreChannelMatch(channels[5], "Reshet 13 (720p)")).toBe(0);
  });

  it("matches COPS exactly", () => {
    const ch = findBestChannelMatch(channels, "COPS");
    expect(ch?.id).toBe("mjh-10-cops");
  });

  it("uses stream path as extra hint", () => {
    const ch = findBestChannelMatch(
      channels,
      "Gravitas Movies (1080p)",
      undefined,
      "https://d6dg3ebeih71x.cloudfront.net/Gravitas_Movies.m3u8",
    );
    expect(ch?.name).toBe("Gravitas Movies");
  });

  it("does not match master.m3u8 streams to Bassmaster", () => {
    const ch = findBestChannelMatch(
      [{ id: "bass", name: "Bassmaster", sourceKey: "roku" }],
      "Entertainment Tonight (1080p)",
      "EntertainmentTonight.us@SD",
      "https://enterbcef94b.airspace-cdn.cbsivideo.com/master.m3u8",
    );
    expect(ch).toBeNull();
  });

  it("does not match ABC Kids to ABC News Live via single token", () => {
    const ch = findBestChannelMatch(
      [
        { id: "abc-news", name: "ABC News Live", sourceKey: "roku" },
        { id: "abc-kids", name: "ABC Kids", sourceKey: "roku" },
      ],
      "ABC Kids",
      "ABCKids.au@Sydney",
    );
    expect(ch?.name).toBe("ABC Kids");
  });

  it("matches ION Mystery via amagi stream path when IPTV name is WFXT-DT2", () => {
    const ch = findBestChannelMatch(
      [{ id: "plex-ion", name: "Ion Mystery", sourceKey: "plex" }],
      "WFXT-DT2 (1080p)",
      "WFXT662.us@SD",
      "https://cdn-uw2-prod.tsv2.amagi.tv/linear/amg01438-ewscrippscompan-ionmystery-tablo/playlist.m3u8",
    );
    expect(ch?.name).toBe("Ion Mystery");
  });
});
