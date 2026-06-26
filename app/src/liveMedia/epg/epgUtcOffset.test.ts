import { describe, expect, it } from "vitest";
import type { HlsCueState } from "./hlsCueSync";
import type { EpgProgram } from "./types";
import { detectEpgOffsetHoursForCue, shiftEpgPrograms } from "./epgUtcOffset";
import { findLiveProgram, findProgramForStreamCue } from "./epgProgramSync";

function prog(title: string, start: string, end: string): EpgProgram {
  return {
    channelId: "c1",
    title,
    start: new Date(start),
    end: new Date(end),
  };
}

describe("epgUtcOffset", () => {
  it("detects +4h offset when EPG times are US-local mislabeled as UTC", () => {
    const now = new Date("2026-06-26T03:30:00Z");
    const cue: HlsCueState = { elapsedMinutes: 30, durationMinutes: 60 };
    const programs = [prog("Moon Landing", "2026-06-25T23:00:00Z", "2026-06-26T00:00:00Z")];

    expect(findLiveProgram(programs, now)).toBeNull();
    const offset = detectEpgOffsetHoursForCue(programs, cue, now);
    expect(offset).toBe(4);
    const shifted = shiftEpgPrograms(programs, offset);
    expect(findProgramForStreamCue(shifted, now, cue)?.title).toBe("Moon Landing");
  });
});
