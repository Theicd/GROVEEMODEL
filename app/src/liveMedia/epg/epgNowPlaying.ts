import type { EpgGuideEntry } from "./epgGuideStore";
import {
  progressFromWindow,
  resolveLiveEpgProgram,
  resolveProgramWindow,
} from "./epgProgramSync";
import type { HlsCueState } from "./hlsCueSync";
import type { EpgProgram } from "./types";

export { findLiveProgram, resolveLiveEpgProgram } from "./epgProgramSync";

export type NowPlayingInfo = {
  program: EpgProgram;
  progressPct: number;
  minutesLeft: number;
  /** Times shown in OSD — stream-synced when HLS cues are available. */
  displayStart: Date;
  displayEnd: Date;
  durationMinutes: number;
  streamSynced: boolean;
};

export function nowPlayingFromEntry(
  entry: EpgGuideEntry | null | undefined,
  now = new Date(),
  cue?: HlsCueState | null,
  movieRuntimeMinutes?: number | null,
): NowPlayingInfo | null {
  const programs = entry?.schedule?.programs ?? [];
  const streamUrl = entry?.hit?.mediaPlayUrl || entry?.hit?.url;
  const sourceKey = entry?.schedule?.channel?.sourceKey;
  const tvgId = typeof entry?.hit?.meta?.tvgId === "string" ? entry.hit.meta.tvgId : undefined;
  const program = resolveLiveEpgProgram(programs, now, { cue, streamUrl, sourceKey, tvgId });
  if (!program) return null;

  const { start: displayStart, end: displayEnd, durationMinutes } = resolveProgramWindow(
    program,
    now,
    cue,
    movieRuntimeMinutes,
  );
  const { progressPct, minutesLeft } = progressFromWindow(displayStart, displayEnd, now);

  return {
    program,
    progressPct,
    minutesLeft,
    displayStart,
    displayEnd,
    durationMinutes: Math.max(1, Math.round(durationMinutes)),
    streamSynced: cue != null,
  };
}

export function entryForHit(entries: EpgGuideEntry[], hitId: string | undefined): EpgGuideEntry | null {
  if (!hitId) return null;
  return entries.find((e) => e.hit.id === hitId) ?? null;
}

export function formatOsdProgramTime(d: Date, rtl: boolean): string {
  return d.toLocaleTimeString(rtl ? "he-IL" : "en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: !rtl,
  });
}

export function formatOsdProgramRange(start: Date, end: Date, rtl: boolean): string {
  return `${formatOsdProgramTime(start, rtl)} – ${formatOsdProgramTime(end, rtl)}`;
}

export function formatEpisodeLabel(program: EpgProgram): string | null {
  if (program.season != null && program.episode != null) {
    return `S${String(program.season).padStart(2, "0")}E${String(program.episode).padStart(2, "0")}`;
  }
  if (program.episode != null) return `Ep. ${program.episode}`;
  if (program.episodeLabel?.trim()) return program.episodeLabel.trim();
  return null;
}

export function formatOsdDescription(program: EpgProgram, maxLen = 140): string | null {
  const text = program.description?.trim() || program.subTitle?.trim();
  if (!text) return null;
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1).trim()}…`;
}

export function formatDurationMinutes(mins: number, rtl: boolean): string {
  const n = Math.max(1, Math.round(mins));
  return rtl ? `${n} דק׳` : `${n} min`;
}
