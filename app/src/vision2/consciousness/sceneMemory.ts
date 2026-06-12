/** Scene timeline — HAL reacts to transitions, not frames. */

import type { SceneMemoryEntry, SoulState } from "./types";

const MAX_ENTRIES = 32;

export class SceneMemory {
  private entries: SceneMemoryEntry[] = [];

  reset(): void {
    this.entries = [];
  }

  push(entry: SceneMemoryEntry): void {
    const last = this.entries[this.entries.length - 1];
    if (last && last.soul === entry.soul && last.transition === entry.transition) return;
    this.entries.push(entry);
    if (this.entries.length > MAX_ENTRIES) this.entries.shift();
  }

  recent(limit = 8): SceneMemoryEntry[] {
    return this.entries.slice(-limit);
  }

  lastTransition(): SceneMemoryEntry | null {
    return this.entries[this.entries.length - 1] ?? null;
  }

  evolutionLine(): string {
    const souls = this.recent(6).map((e) => e.soul.replace(/_/g, " "));
    if (!souls.length) return "VOID_IDLE";
    return souls.join(" → ");
  }

  transitionsSince(soul: SoulState, windowSec: number, now: number): number {
    return this.entries.filter((e) => e.soul === soul && now - e.t <= windowSec * 1000).length;
  }
}
