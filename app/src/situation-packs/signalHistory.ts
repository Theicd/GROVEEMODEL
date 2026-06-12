/** Rolling signal history for duration / repetition / silence. */

export type SignalHistory = {
  /** signal key → timestamps in window */
  hits: Map<string, number[]>;
  lastMotionAt: number;
  lastInteractionAt: number;
  personPresentSince: number;
};

export const createSignalHistory = (): SignalHistory => ({
  hits: new Map(),
  lastMotionAt: 0,
  lastInteractionAt: 0,
  personPresentSince: 0,
});

const prune = (times: number[], windowSec: number, now: number): number[] =>
  times.filter((t) => now - t <= windowSec * 1000);

export const recordSignal = (
  history: SignalHistory,
  key: string,
  now = Date.now(),
  windowSec = 30,
): void => {
  const prev = history.hits.get(key) ?? [];
  history.hits.set(key, prune([...prev, now], windowSec, now));
  history.lastInteractionAt = now;
};

export const signalCount = (
  history: SignalHistory,
  key: string,
  windowSec: number,
  now = Date.now(),
): number => {
  const times = history.hits.get(key) ?? [];
  return prune(times, windowSec, now).length;
};

export const signalDurationSec = (
  history: SignalHistory,
  key: string,
  now = Date.now(),
): number => {
  const times = history.hits.get(key) ?? [];
  if (!times.length) return 0;
  const first = Math.min(...times);
  const last = Math.max(...times);
  if (times.length === 1) return (now - first) / 1000;
  return (last - first) / 1000;
};

export const updateMotionHistory = (
  history: SignalHistory,
  motionLevel: number,
  personPresent: boolean,
  now = Date.now(),
): void => {
  if (motionLevel >= 0.08) history.lastMotionAt = now;
  if (personPresent && !history.personPresentSince) history.personPresentSince = now;
  if (!personPresent) history.personPresentSince = 0;
};

export const silenceSec = (history: SignalHistory, now = Date.now()): number => {
  if (!history.lastInteractionAt) return 0;
  return Math.max(0, (now - history.lastInteractionAt) / 1000);
};

export const countInteractionSignals = (
  history: SignalHistory,
  windowSec: number,
  now = Date.now(),
): number => {
  let total = 0;
  for (const [key, times] of history.hits) {
    if (!key.startsWith("gesture:") && !key.startsWith("event:") && !key.startsWith("body:")) continue;
    total += prune(times, windowSec, now).length;
  }
  return total;
};

export const resetSignalHistory = (history: SignalHistory): void => {
  history.hits.clear();
  history.lastMotionAt = 0;
  history.lastInteractionAt = 0;
  history.personPresentSince = 0;
};
