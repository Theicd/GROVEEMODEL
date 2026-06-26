import { describe, expect, it } from "vitest";
import { findLiveProgram, nowPlayingFromEntry } from "./epgNowPlaying";
import type { EpgGuideEntry } from "./epgGuideStore";

describe("epgNowPlaying", () => {
  it("finds the programme on air now", () => {
    const now = new Date("2026-06-25T22:30:00Z");
    const programs = [
      {
        channelId: "c1",
        title: "Past",
        start: new Date("2026-06-25T21:00:00Z"),
        end: new Date("2026-06-25T22:00:00Z"),
      },
      {
        channelId: "c1",
        title: "Live Show",
        start: new Date("2026-06-25T22:00:00Z"),
        end: new Date("2026-06-25T23:00:00Z"),
      },
    ];
    expect(findLiveProgram(programs, now)?.title).toBe("Live Show");
  });

  it("builds progress for OSD", () => {
    const now = new Date("2026-06-25T22:30:00Z");
    const entry: EpgGuideEntry = {
      hit: { id: "livetv-x", kind: "livetv", title: "COPS", url: "", snippet: "" },
      schedule: {
        channel: { id: "c1", name: "COPS", sourceKey: "mjh-all" },
        sourceLabel: "MJH",
        programs: [
          {
            channelId: "c1",
            title: "On Patrol",
            start: new Date("2026-06-25T22:00:00Z"),
            end: new Date("2026-06-25T23:00:00Z"),
          },
        ],
      },
    };
    const info = nowPlayingFromEntry(entry, now);
    expect(info?.program.title).toBe("On Patrol");
    expect(info?.progressPct).toBe(50);
    expect(info?.minutesLeft).toBe(30);
    expect(info?.durationMinutes).toBe(60);
    expect(info?.streamSynced).toBe(false);
  });
});
