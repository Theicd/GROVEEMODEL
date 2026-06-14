import type { LiveWorldSnapshot } from "./types";

const SNAPSHOT_TTL_MS = 90_000;

let cached: LiveWorldSnapshot | null = null;
let inflight: Promise<LiveWorldSnapshot | null> | null = null;

export function getCachedLiveWorldSnapshot(maxAgeMs = SNAPSHOT_TTL_MS): LiveWorldSnapshot | null {
  if (!cached) return null;
  if (Date.now() - cached.fetchedAt > maxAgeMs) return null;
  return cached;
}

export function setLiveWorldSnapshot(snapshot: LiveWorldSnapshot): void {
  cached = snapshot;
}

export function mergeLiveWorldSnapshot(partial: Partial<LiveWorldSnapshot>): LiveWorldSnapshot {
  const base = cached ?? { fetchedAt: Date.now(), source: "mixed" as const };
  cached = {
    ...base,
    ...partial,
    fetchedAt: partial.fetchedAt ?? Date.now(),
    source: partial.source ?? base.source ?? "mixed",
  };
  return cached;
}

export function getInflightSnapshotFetch(): Promise<LiveWorldSnapshot | null> | null {
  return inflight;
}

export function setInflightSnapshotFetch(p: Promise<LiveWorldSnapshot | null> | null): void {
  inflight = p;
}

export function clearLiveWorldSnapshotCache(): void {
  cached = null;
  inflight = null;
}
