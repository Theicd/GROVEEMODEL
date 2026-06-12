/** L7 — episodic memory (understandings + durations, not raw detections). */

import type { BodyLanguageVector, EpisodicEntry, EpisodicEntryKind } from "./types";

const MAX_ENTRIES = 24;
const ENTRY_TTL_MS = 30 * 60 * 1000;

export class EpisodicMemory {
  private entries: EpisodicEntry[] = [];
  private openEpisodes = new Map<EpisodicEntryKind, number>();

  reset(): void {
    this.entries = [];
    this.openEpisodes.clear();
  }

  tickEpisode(
    kind: EpisodicEntryKind,
    active: boolean,
    peak?: Partial<BodyLanguageVector>,
    now = Date.now(),
  ): void {
    if (active) {
      if (!this.openEpisodes.has(kind)) {
        this.openEpisodes.set(kind, now);
      }
      return;
    }
    const startedAt = this.openEpisodes.get(kind);
    if (!startedAt) return;
    this.openEpisodes.delete(kind);
    const durationSec = Math.max(1, Math.floor((now - startedAt) / 1000));
    this.push({
      kind,
      startedAt,
      durationSec,
      peakScores: peak,
    });
  }

  recordInstant(kind: EpisodicEntryKind, now = Date.now()): void {
    this.push({ kind, startedAt: now, durationSec: 0 });
  }

  private push(entry: EpisodicEntry): void {
    this.entries.unshift(entry);
    const cutoff = Date.now() - ENTRY_TTL_MS;
    this.entries = this.entries.filter((e) => e.startedAt >= cutoff).slice(0, MAX_ENTRIES);
  }

  summarize(max = 4): string[] {
    const lines: string[] = [];
    for (const e of this.entries.slice(0, max)) {
      if (e.kind === "greeting") {
        lines.push("User greeting detected recently.");
      } else if (e.kind === "focus_block") {
        lines.push(`Focused work block ~${e.durationSec}s.`);
      } else if (e.kind === "stress_episode") {
        lines.push(`Stress episode ~${e.durationSec}s.`);
      } else if (e.kind === "face_touch") {
        lines.push(`Face touch / thinking posture ~${e.durationSec}s.`);
      } else if (e.kind === "break") {
        lines.push(`Break / drink moment recorded.`);
      }
    }
    return lines;
  }

  getEntries(): EpisodicEntry[] {
    return [...this.entries];
  }
}
