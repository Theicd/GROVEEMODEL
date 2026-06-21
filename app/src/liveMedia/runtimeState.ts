import type { Channel, Source, StreamStatus } from "./types";

export type LiveMediaPhase = "idle" | "syncing" | "validating";

export interface LiveMediaProgress {
  phase: LiveMediaPhase;
  current: number;
  total: number;
  label: string;
}

export interface LiveMediaStatusBreakdown {
  working: number;
  warning: number;
  offline: number;
  unknown: number;
}

export interface LiveMediaCategoryCount {
  category: string;
  count: number;
}

export interface LiveMediaCatalogSummary {
  channels: number;
  radio: number;
  channelStatus: LiveMediaStatusBreakdown;
  radioStatus: LiveMediaStatusBreakdown;
  categories: LiveMediaCategoryCount[];
  sources: Source[];
  lastSyncAt: number | null;
  progress: LiveMediaProgress;
  lastError: string | null;
}

const emptyBreakdown = (): LiveMediaStatusBreakdown => ({
  working: 0,
  warning: 0,
  offline: 0,
  unknown: 0,
});

const idleProgress = (): LiveMediaProgress => ({
  phase: "idle",
  current: 0,
  total: 0,
  label: "",
});

let progress: LiveMediaProgress = idleProgress();
let lastError: string | null = null;
const listeners = new Set<(summary: LiveMediaCatalogSummary) => void>();

export function subscribeLiveMediaSummary(cb: (summary: LiveMediaCatalogSummary) => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function setLiveMediaProgress(next: LiveMediaProgress): void {
  progress = next;
  notifyLiveMediaSummary();
}

export function setLiveMediaError(message: string | null): void {
  lastError = message;
  notifyLiveMediaSummary();
}

export function getLiveMediaProgress(): LiveMediaProgress {
  return progress;
}

export function getLiveMediaLastError(): string | null {
  return lastError;
}

export function statusBreakdown(items: Array<{ status?: StreamStatus }>): LiveMediaStatusBreakdown {
  const out = emptyBreakdown();
  for (const item of items) {
    const s = item.status ?? "unknown";
    if (s === "working") out.working += 1;
    else if (s === "warning") out.warning += 1;
    else if (s === "offline") out.offline += 1;
    else out.unknown += 1;
  }
  return out;
}

export function categoryBreakdown(channels: Channel[]): LiveMediaCategoryCount[] {
  const map = new Map<string, number>();
  for (const c of channels) {
    const cat = c.category || "general";
    map.set(cat, (map.get(cat) ?? 0) + 1);
  }
  return [...map.entries()]
    .map(([category, count]) => ({ category, count }))
    .sort((a, b) => b.count - a.count);
}

export function notifyLiveMediaSummary(partial?: Partial<LiveMediaCatalogSummary>): void {
  if (partial) {
    for (const cb of listeners) cb(partial as LiveMediaCatalogSummary);
    return;
  }
  void import("./catalogStore").then(({ buildLiveMediaCatalogSummary }) => {
    void buildLiveMediaCatalogSummary().then((summary) => {
      for (const cb of listeners) cb(summary);
    });
  });
}

export function resetLiveMediaProgress(): void {
  progress = idleProgress();
}
