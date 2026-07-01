import type { UnifiedSearchHit } from "../searchResults/types";

/**
 * Persistent, self-healing channel health store.
 *
 * Goals:
 *  - Open the tuner on *different* channels each visit (weighted-random lineup).
 *  - Prefer channels that historically go live fast and reliably.
 *  - Quietly drop channels that keep failing, so the viewer never sees dead tiles
 *    and never has to reload the page to "fix" the lineup.
 *
 * Everything is stored in localStorage and degrades gracefully if unavailable.
 */

const STORE_KEY = "lm-cable-health-v1";
/** Consecutive-ish failures (with no successes) before a channel is hidden. */
const KNOWN_BAD_FAILS = 3;
/** A "known bad" verdict expires after this, giving the channel another chance. */
const KNOWN_BAD_TTL_MS = 3 * 24 * 60 * 60 * 1000;
/** Clamp measured go-live times into a sane band for weighting. */
const MS_FLOOR = 500;
const MS_CEIL = 9000;

type HealthRec = {
  /** Successful go-live count. */
  ok: number;
  /** Failure count (never reached playable state in time). */
  fail: number;
  /** Exponentially-smoothed time-to-first-frame, ms. */
  ms: number;
  /** Last update epoch ms. */
  ts: number;
};

type HealthStore = Record<string, HealthRec>;

let cache: HealthStore | null = null;

function load(): HealthStore {
  if (cache) return cache;
  try {
    const raw = localStorage.getItem(STORE_KEY);
    cache = raw ? (JSON.parse(raw) as HealthStore) : {};
  } catch {
    cache = {};
  }
  return cache;
}

let saveTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleSave() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(cache ?? {}));
    } catch {
      /* storage full / unavailable — ignore */
    }
  }, 400);
}

/** Stable key for a channel across sessions (id first, then stream URL). */
export function channelKeyOf(hit: UnifiedSearchHit | null | undefined): string {
  if (!hit) return "";
  return hit.id || hit.mediaPlayUrl || hit.url || "";
}

export function recordChannelReady(hit: UnifiedSearchHit | null | undefined, elapsedMs: number): void {
  const key = channelKeyOf(hit);
  if (!key) return;
  const store = load();
  const prev = store[key];
  const clamped = Math.max(MS_FLOOR, Math.min(MS_CEIL, Math.round(elapsedMs) || MS_CEIL));
  const ms = prev?.ms ? Math.round(prev.ms * 0.6 + clamped * 0.4) : clamped;
  store[key] = {
    ok: (prev?.ok ?? 0) + 1,
    // A success heals one prior failure, so temporarily-flaky channels recover.
    fail: Math.max(0, (prev?.fail ?? 0) - 1),
    ms,
    ts: Date.now(),
  };
  scheduleSave();
}

export function recordChannelFail(hit: UnifiedSearchHit | null | undefined): void {
  const key = channelKeyOf(hit);
  if (!key) return;
  const store = load();
  const prev = store[key];
  store[key] = {
    ok: prev?.ok ?? 0,
    fail: (prev?.fail ?? 0) + 1,
    ms: prev?.ms ?? MS_CEIL,
    ts: Date.now(),
  };
  scheduleSave();
}

function isRecentlyBad(rec: HealthRec | undefined): boolean {
  if (!rec) return false;
  if (rec.ok > 0) return false;
  if (rec.fail < KNOWN_BAD_FAILS) return false;
  return Date.now() - rec.ts < KNOWN_BAD_TTL_MS;
}

export function isChannelKnownBad(hit: UnifiedSearchHit | null | undefined): boolean {
  return isRecentlyBad(load()[channelKeyOf(hit)]);
}

/** Relative selection weight — higher means more likely to be featured on open. */
function weightForKey(key: string): number {
  const rec = load()[key];
  if (!rec) return 1; // untested: fair chance so the lineup keeps exploring
  if (isRecentlyBad(rec)) return 0; // hidden entirely
  if (rec.ok === 0) return 0.12; // failed before but not yet condemned — rare retry
  const reliability = rec.ok / (rec.ok + rec.fail); // 0..1
  const speed = MS_CEIL / Math.max(MS_FLOOR, rec.ms); // fast → >1, slow → <1
  const speedFactor = Math.max(0.3, Math.min(3, speed));
  return Math.max(0.05, reliability) * speedFactor;
}

/** Favorite indices that should be hidden from the viewer right now. */
export function knownBadFavoriteSet(favorites: UnifiedSearchHit[]): Set<number> {
  const bad = new Set<number>();
  favorites.forEach((hit, i) => {
    if (isChannelKnownBad(hit)) bad.add(i);
  });
  return bad;
}

/** Weighted-random sample of `count` distinct favorite indices (fresh each call). */
export function pickHealthyQuadSlots(favorites: UnifiedSearchHit[], count = 4): number[] {
  const total = favorites.length;
  if (total <= 0) return Array.from({ length: count }, () => 0);
  if (total <= count) return Array.from({ length: count }, (_, i) => i % total);

  let pool = favorites
    .map((hit, i) => ({ i, w: weightForKey(channelKeyOf(hit)) }))
    .filter((c) => c.w > 0);

  // If almost everything is condemned, fall back to the full list so we still fill the grid.
  if (pool.length < count) {
    pool = favorites.map((hit, i) => ({ i, w: Math.max(0.05, weightForKey(channelKeyOf(hit)) || 0.5) }));
  }

  const chosen: number[] = [];
  const working = [...pool];
  while (chosen.length < count && working.length > 0) {
    const totalW = working.reduce((s, c) => s + c.w, 0);
    let r = Math.random() * totalW;
    let idx = 0;
    for (let k = 0; k < working.length; k += 1) {
      r -= working[k].w;
      if (r <= 0) {
        idx = k;
        break;
      }
    }
    chosen.push(working[idx].i);
    working.splice(idx, 1);
  }
  while (chosen.length < count) chosen.push(chosen[chosen.length - 1] ?? 0);
  return chosen;
}
