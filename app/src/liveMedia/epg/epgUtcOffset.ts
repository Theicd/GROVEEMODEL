import type { HlsCueState } from "./hlsCueSync";
import { cueWindow } from "./hlsCueSync";
import type { EpgProgram } from "./types";

const streamOffsetCache = new Map<string, number>();

const US_MJH_SOURCES = new Set([
  "mjh-plex-us",
  "mjh-pluto-us",
  "mjh-samsung-us",
  "mjh-roku",
]);

export function isUsMjhEpgSource(sourceKey?: string): boolean {
  return sourceKey != null && US_MJH_SOURCES.has(sourceKey);
}

export function shiftEpgPrograms(programs: EpgProgram[], offsetHours: number): EpgProgram[] {
  if (!offsetHours) return programs;
  const ms = offsetHours * 3_600_000;
  return programs.map((p) => ({
    ...p,
    start: new Date(p.start.getTime() + ms),
    end: new Date(p.end.getTime() + ms),
  }));
}

export function cacheStreamEpgOffset(streamUrl: string, offsetHours: number): void {
  if (!streamUrl) return;
  streamOffsetCache.set(streamUrl, offsetHours);
}

export function getStreamEpgOffset(streamUrl?: string): number {
  if (!streamUrl) return 0;
  return streamOffsetCache.get(streamUrl) ?? 0;
}

export function resetStreamEpgOffsetCacheForTests(): void {
  streamOffsetCache.clear();
}

/**
 * MJH US feeds often stamp local US wall time with +0000.
 * Slide EPG slots until they best overlap the stream's HLS cue window.
 */
export function detectEpgOffsetHoursForCue(
  programs: EpgProgram[],
  cue: HlsCueState,
  now: Date,
): number {
  const { start: winStart, end: winEnd } = cueWindow(now, cue);
  const winStartMs = winStart.getTime();
  const winEndMs = winEnd.getTime();
  const winSpan = Math.max(winEndMs - winStartMs, 60_000);

  let bestOffset = 0;
  let bestScore = -Infinity;

  for (let offset = -12; offset <= 12; offset++) {
    const shift = offset * 3_600_000;
    for (const program of programs) {
      const s = program.start.getTime() + shift;
      const e = program.end.getTime() + shift;
      const overlap = Math.max(0, Math.min(e, winEndMs) - Math.max(s, winStartMs));
      if (overlap < 90_000) continue;

      const overlapRatio = overlap / winSpan;
      const slotMs = Math.max(e - s, 1);
      const durDelta = Math.abs(slotMs - cue.durationMinutes * 60_000);
      const startDelta = Math.abs(s - winStartMs);
      const score =
        overlapRatio * 2000 -
        durDelta / 90_000 -
        startDelta / 180_000 -
        Math.abs(slotMs - winSpan) / 600_000;

      if (score > bestScore) {
        bestScore = score;
        bestOffset = offset;
      }
    }
  }

  return bestScore > 400 ? bestOffset : 0;
}

/** When no HLS cue yet, pick smallest offset that yields a plausible live slot. */
export function inferUsEpgOffsetHours(programs: EpgProgram[], now: Date): number {
  let bestOffset = 0;
  let bestScore = -Infinity;

  for (let offset = -8; offset <= 8; offset++) {
    const shifted = shiftEpgPrograms(programs, offset);
    const live = shifted.find((p) => p.start <= now && p.end > now);
    if (!live) continue;
    const slotMin = (live.end.getTime() - live.start.getTime()) / 60_000;
    if (slotMin < 12 || slotMin > 300) continue;
    const score = 200 - Math.abs(offset) * 12 - Math.abs(slotMin - 60) * 0.5;
    if (score > bestScore) {
      bestScore = score;
      bestOffset = offset;
    }
  }

  return bestScore > 0 ? bestOffset : 0;
}

/** @deprecated use inferUsEpgOffsetHours */
export function guessUsEpgOffsetHours(programs: EpgProgram[], now: Date): number {
  return inferUsEpgOffsetHours(programs, now);
}

export function resolveEpgOffsetHours(
  programs: EpgProgram[],
  now: Date,
  opts?: { cue?: HlsCueState | null; streamUrl?: string; sourceKey?: string },
): number {
  const { cue, streamUrl, sourceKey } = opts ?? {};
  if (cue) {
    const detected = detectEpgOffsetHoursForCue(programs, cue, now);
    if (detected) {
      if (streamUrl) cacheStreamEpgOffset(streamUrl, detected);
      return detected;
    }
  }

  const cached = getStreamEpgOffset(streamUrl);
  if (cached) return cached;

  if (isUsMjhEpgSource(sourceKey) && programs.length > 0) {
    const guessed = inferUsEpgOffsetHours(programs, now);
    if (guessed && streamUrl) cacheStreamEpgOffset(streamUrl, guessed);
    return guessed;
  }

  return 0;
}
