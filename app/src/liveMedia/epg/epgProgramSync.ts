import type { HlsCueState } from "./hlsCueSync";
import { cueWindow } from "./hlsCueSync";
import {
  resolveEpgOffsetHours,
  shiftEpgPrograms,
} from "./epgUtcOffset";
import type { EpgProgram } from "./types";

export function findLiveProgram(programs: EpgProgram[], now = new Date()): EpgProgram | null {
  return programs.find((p) => p.start <= now && p.end > now) ?? null;
}

const MIN_CUE_MATCH_SCORE = 320;

/** Match EPG title to the stream's real programme window (from HLS cues), not wall-clock slot only. */
export function findProgramForStreamCue(
  programs: EpgProgram[],
  now: Date,
  cue: HlsCueState,
): EpgProgram | null {
  if (!programs.length) return null;

  const { start: winStart, end: winEnd } = cueWindow(now, cue);
  const winStartMs = winStart.getTime();
  const winEndMs = winEnd.getTime();
  const winSpan = winEndMs - winStartMs;

  let best: { program: EpgProgram; score: number } | null = null;

  for (const program of programs) {
    const oStart = Math.max(program.start.getTime(), winStartMs);
    const oEnd = Math.min(program.end.getTime(), winEndMs);
    const overlap = Math.max(0, oEnd - oStart);
    if (overlap < 3 * 60_000) continue;

    const overlapRatio = winSpan > 0 ? overlap / winSpan : 0;
    const startDelta = Math.abs(program.start.getTime() - winStartMs);
    const durationDelta = Math.abs(
      program.end.getTime() - program.start.getTime() - cue.durationMinutes * 60_000,
    );
    const score = overlapRatio * 1000 - startDelta / 60_000 - durationDelta / 120_000;

    if (!best || score > best.score) best = { program, score };
  }

  if (!best || best.score < MIN_CUE_MATCH_SCORE) return null;
  return best.program;
}

export function resolveLiveEpgProgram(
  programs: EpgProgram[],
  now: Date,
  opts?: { cue?: HlsCueState | null; streamUrl?: string; sourceKey?: string },
): EpgProgram | null {
  if (!programs.length) return null;
  const offset = resolveEpgOffsetHours(programs, now, opts);
  const shifted = shiftEpgPrograms(programs, offset);
  if (opts?.cue) {
    const fromCue = findProgramForStreamCue(shifted, now, opts.cue);
    if (fromCue) return fromCue;
  }
  return findLiveProgram(shifted, now);
}

export function effectiveProgramDurationMinutes(program: EpgProgram, cue?: HlsCueState | null): number {
  if (cue) return cue.durationMinutes;
  if (program.lengthMinutes != null && program.lengthMinutes > 0) {
    const slot = (program.end.getTime() - program.start.getTime()) / 60_000;
    if (program.lengthMinutes < slot - 2) return program.lengthMinutes;
  }
  return (program.end.getTime() - program.start.getTime()) / 60_000;
}

export function resolveProgramWindow(
  program: EpgProgram,
  now: Date,
  cue?: HlsCueState | null,
  movieRuntimeMinutes?: number | null,
): { start: Date; end: Date; durationMinutes: number } {
  if (cue) {
    const { start, end: blockEnd } = cueWindow(now, cue);
    const blockMins = cue.durationMinutes;
    const contentMins =
      movieRuntimeMinutes != null && movieRuntimeMinutes > 0 && movieRuntimeMinutes < blockMins - 2
        ? movieRuntimeMinutes
        : blockMins;
    const startMs = start.getTime();
    return {
      start,
      end: new Date(startMs + contentMins * 60_000),
      durationMinutes: contentMins,
    };
  }

  const slotStart = program.start.getTime();
  const durationMs = effectiveProgramDurationMinutes(program, null) * 60_000;
  const contentEnd = Math.min(slotStart + durationMs, program.end.getTime());
  const mins = (contentEnd - slotStart) / 60_000;
  return { start: program.start, end: new Date(contentEnd), durationMinutes: mins };
}

export function progressFromWindow(
  displayStart: Date,
  displayEnd: Date,
  now: Date,
): { progressPct: number; minutesLeft: number } {
  const span = displayEnd.getTime() - displayStart.getTime();
  const elapsed = now.getTime() - displayStart.getTime();
  const progressPct = span > 0 ? Math.min(100, Math.max(0, (elapsed / span) * 100)) : 0;
  const minutesLeft = Math.max(0, Math.round((displayEnd.getTime() - now.getTime()) / 60_000));
  return { progressPct, minutesLeft };
}
