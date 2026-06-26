import { describe, expect, it } from "vitest";
import { normalizeCueTimes, parseHlsCueFromPlaylist } from "./hlsCueSync";
import { nowPlayingFromEntry } from "./epgNowPlaying";
import type { EpgGuideEntry } from "./epgGuideStore";

describe("hlsCueSync", () => {
  it("parses Gravitas-style cue durations as minutes", () => {
    const text = `#EXTM3U
#EXT-X-CUE-OUT-CONT:ElapsedTime=74.975,Duration=120.053,SCTE35=abc
#EXTINF:6.0,
seg.ts`;
    const cue = parseHlsCueFromPlaylist(text);
    expect(cue).toEqual({ elapsedMinutes: 74.975, durationMinutes: 120.053 });
  });

  it("parses short ad cues as seconds converted to minutes", () => {
    expect(normalizeCueTimes(30, 45)).toEqual({ elapsedMinutes: 0.5, durationMinutes: 0.75 });
  });
});

describe("nowPlaying stream sync", () => {
  it("uses HLS cue window instead of EPG slot end", () => {
    const now = new Date("2026-06-25T23:30:00Z");
    const entry: EpgGuideEntry = {
      hit: { id: "g1", kind: "livetv", title: "Gravitas", url: "", snippet: "" },
      schedule: {
        channel: { id: "c1", name: "Gravitas Movies", sourceKey: "samsung" },
        sourceLabel: "Samsung",
        programs: [
          {
            channelId: "c1",
            title: "Eye See You",
            start: new Date("2026-06-25T21:17:00Z"),
            end: new Date("2026-06-25T23:09:00Z"),
          },
        ],
      },
    };
    const cue = { elapsedMinutes: 90, durationMinutes: 120 };
    const info = nowPlayingFromEntry(entry, now, cue);
    expect(info?.program.title).toBe("Eye See You");
    expect(info?.durationMinutes).toBe(120);
    expect(info?.displayStart.toISOString()).toBe("2026-06-25T22:00:00.000Z");
    expect(info?.displayEnd.toISOString()).toBe("2026-06-26T00:00:00.000Z");
    expect(info?.minutesLeft).toBe(30);
    expect(info?.streamSynced).toBe(true);
  });

  it("shortens display to TMDB runtime inside stream block", () => {
    const now = new Date("2026-06-25T22:30:00Z");
    const entry: EpgGuideEntry = {
      hit: { id: "g1", kind: "livetv", title: "Gravitas", url: "", snippet: "" },
      schedule: {
        channel: { id: "c1", name: "Gravitas Movies", sourceKey: "samsung" },
        sourceLabel: "Samsung",
        programs: [
          {
            channelId: "c1",
            title: "Kill Switch",
            start: new Date("2026-06-25T21:00:00Z"),
            end: new Date("2026-06-25T23:30:00Z"),
          },
        ],
      },
    };
    const cue = { elapsedMinutes: 45, durationMinutes: 120 };
    const info = nowPlayingFromEntry(entry, now, cue, 89);
    expect(info?.durationMinutes).toBe(89);
    expect(info?.displayEnd.toISOString()).toBe("2026-06-25T23:14:00.000Z");
    expect(info?.minutesLeft).toBe(44);
  });
});
