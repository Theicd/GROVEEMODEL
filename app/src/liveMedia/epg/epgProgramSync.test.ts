import { describe, expect, it } from "vitest";
import type { HlsCueState } from "./hlsCueSync";
import type { EpgProgram } from "./types";
import {
  findProgramNearestCueStart,
  findProgramForStreamCue,
  resolveLiveEpgProgram,
} from "./epgProgramSync";

function prog(title: string, start: string, end: string): EpgProgram {
  return {
    channelId: "c1",
    title,
    start: new Date(start),
    end: new Date(end),
  };
}

describe("epgProgramSync cue fallback", () => {
  it("findProgramNearestCueStart matches cue slot when strict overlap is too small", () => {
    const now = new Date("2026-06-26T20:10:00Z");
    const cue: HlsCueState = { elapsedMinutes: 10, durationMinutes: 60 };
    const programs = [prog("On Stream", "2026-06-26T19:58:00Z", "2026-06-26T20:01:00Z")];

    expect(findProgramForStreamCue(programs, now, cue)).toBeNull();
    expect(findProgramNearestCueStart(programs, now, cue)?.title).toBe("On Stream");
  });

  it("resolveLiveEpgProgram uses nearest cue slot instead of returning null", () => {
    const now = new Date("2026-06-26T20:10:00Z");
    const cue: HlsCueState = { elapsedMinutes: 10, durationMinutes: 60 };
    const programs = [prog("On Stream", "2026-06-26T19:58:00Z", "2026-06-26T20:01:00Z")];

    expect(resolveLiveEpgProgram(programs, now, { cue })?.title).toBe("On Stream");
  });
});
