import type { UnifiedSearchHit } from "../searchResults/types";

export type CableTile = UnifiedSearchHit | null;

const QUAD_TILES = 4;
/** Spread initial quad tiles across the lineup (not 1–4 sequential). */
const QUAD_SPREAD_FRACS = [0.35, 0.455, 0.718, 0.963];
export const QUAD_ROTATE_MS = 30_000;
/** If a tile never reaches playable state, skip to the next favorite. */
export const CABLE_STREAM_LOAD_MS = 11_000;
export const CABLE_WARM_SWITCH_MS = 350;

/** First page only: four favorites in a 2×2 grid (rotating). */
export function isQuadPage(pageIndex: number): boolean {
  return pageIndex === 0;
}

/** Page 0 = quad; pages 1..total = one favorite each (indices 0..total-1). */
export function maxCablePageIndex(total: number): number {
  if (total <= 0) return 0;
  return total;
}

export function nextCablePageIndex(current: number, delta: 1 | -1, total: number): number {
  const max = maxCablePageIndex(total);
  let next = current + delta;
  if (next > max) next = 0;
  if (next < 0) next = max;
  return next;
}

/** Favorites array index for a single-channel page (page ≥ 1). */
export function singleFavoriteIndex(pageIndex: number): number {
  return pageIndex - 1;
}

/** 1-based channel number shown in the OSD for single pages. */
export function cableChannelNumber(pageIndex: number): number {
  if (isQuadPage(pageIndex)) return 0;
  return singleFavoriteIndex(pageIndex) + 1;
}

export function cableOsdRangeLabel(pageIndex: number, total: number, quadSlots?: number[]): { from: number; to: number } | null {
  if (!total) return null;
  if (!isQuadPage(pageIndex)) {
    const ch = cableChannelNumber(pageIndex);
    return { from: ch, to: ch };
  }
  if (quadSlots?.length) return quadOsdChannelRange(quadSlots);
  return quadOsdChannelRange(initialQuadSlots(total));
}

/** Evenly spread favorite indices for the opening quad (e.g. ch 20, 26, 40, 53 on a long list). */
export function spreadQuadSlots(total: number): number[] {
  if (total <= 0) return [0, 0, 0, 0];
  if (total <= 4) return Array.from({ length: QUAD_TILES }, (_, i) => i % total);
  const raw = QUAD_SPREAD_FRACS.map((f) => Math.min(total - 1, Math.floor(f * total)));
  const unique: number[] = [];
  for (const idx of raw) {
    let n = idx;
    let guard = 0;
    while (unique.includes(n) && guard < total) {
      n = (n + 1) % total;
      guard += 1;
    }
    unique.push(n);
  }
  const step = quadSpreadStep(total);
  while (unique.length < QUAD_TILES) {
    const seed = unique[unique.length - 1] ?? 0;
    let n = (seed + step) % total;
    let guard = 0;
    while (unique.includes(n) && guard < total) {
      n = (n + 1) % total;
      guard += 1;
    }
    unique.push(n);
  }
  return unique.slice(0, QUAD_TILES);
}

/** Jump size when injecting the next quad channel. */
export function quadSpreadStep(total: number): number {
  if (total <= 4) return 1;
  return Math.max(1, Math.round(total / 5));
}

/** Starting favorite indices for the quad grid (spread, not sequential). */
export function initialQuadSlots(total: number): number[] {
  return spreadQuadSlots(total);
}

/** Index of the next favorite to inject into the quad rotation. */
export function initialRotationCursor(total: number, slots?: number[]): number {
  if (total <= 0) return 0;
  const s = slots ?? spreadQuadSlots(total);
  const step = quadSpreadStep(total);
  const maxIdx = Math.max(...s);
  let cursor = (maxIdx + step) % total;
  const occupied = new Set(s);
  for (let n = 0; n < total; n++) {
    const candidate = (maxIdx + step + n) % total;
    if (!occupied.has(candidate)) {
      cursor = candidate;
      break;
    }
  }
  return cursor;
}

function occupiedExcept(slots: number[], slotToUpdate: number): Set<number> {
  const set = new Set<number>();
  for (let i = 0; i < slots.length; i += 1) {
    if (i !== slotToUpdate) set.add(slots[i]);
  }
  return set;
}

/** Next favorite for a quad tile — must not duplicate another on-screen tile. */
export function pickNextQuadIndex(slots: number[], slotToUpdate: number, cursor: number, total: number): number {
  if (total <= 0) return 0;
  const occupied = occupiedExcept(slots, slotToUpdate);
  for (let n = 0; n < total; n += 1) {
    const candidate = (cursor + n) % total;
    if (!occupied.has(candidate)) return candidate;
  }
  return cursor % total;
}

/** Advance cursor in spread steps, skipping channels still on screen. */
export function nextQuadCursor(slots: number[], injected: number, total: number): number {
  if (total <= 0) return 0;
  const step = quadSpreadStep(total);
  const occupied = new Set(slots);
  for (let n = 1; n <= total; n += 1) {
    const candidate = (injected + step * n) % total;
    if (!occupied.has(candidate)) return candidate;
  }
  return (injected + 1) % total;
}

/** Replace one quad tile; each slot rotates in turn before any repeats on that tile. */
export function advanceQuadRotation(
  slots: number[],
  slotToUpdate: number,
  cursor: number,
  total: number,
): { slots: number[]; cursor: number } {
  if (total <= 0) return { slots, cursor };
  const nextIdx = pickNextQuadIndex(slots, slotToUpdate, cursor, total);
  const nextSlots = [...slots];
  nextSlots[slotToUpdate % QUAD_TILES] = nextIdx;
  return { slots: nextSlots, cursor: nextQuadCursor(nextSlots, nextIdx, total) };
}

export function quadOsdChannelRange(slotIndices: number[]): { from: number; to: number } {
  const nums = slotIndices.map((i) => i + 1).sort((a, b) => a - b);
  return { from: nums[0] ?? 1, to: nums[nums.length - 1] ?? 1 };
}

/** Quad grid hits from favorite index slots — always 4 cells. */
export function pickCableQuadFromSlots(hits: UnifiedSearchHit[], slotIndices: number[]): CableTile[] {
  if (!hits.length) return Array.from({ length: QUAD_TILES }, () => null);
  return Array.from({ length: QUAD_TILES }, (_, i) => {
    const idx = slotIndices[i] ?? i;
    return hits[idx % hits.length] ?? null;
  });
}

export function favoriteForPage(hits: UnifiedSearchHit[], pageIndex: number): UnifiedSearchHit | null {
  if (!hits.length || isQuadPage(pageIndex)) return null;
  const idx = singleFavoriteIndex(pageIndex);
  return hits[idx] ?? null;
}

/** Preload target for the page after `pageIndex`. */
export function preloadPageIndex(pageIndex: number, total: number): number {
  return nextCablePageIndex(pageIndex, 1, total);
}

/** Next favorite to warm while on quad rotation. */
export function quadPreloadFavoriteIndex(cursor: number, total: number): number {
  if (total <= 0) return 0;
  return cursor % total;
}

/** Next favorite in the lineup (wraps). */
export function nextFavoriteIndex(current: number, total: number): number {
  if (total <= 0) return 0;
  return (current + 1) % total;
}

/** Previous favorite in the lineup (wraps). */
export function prevFavoriteIndex(current: number, total: number): number {
  if (total <= 0) return 0;
  return (current - 1 + total) % total;
}

/** Skip favorites marked dead until a working index is found. */
export function nextWorkingFavoriteIndex(
  fromFav: number,
  delta: 1 | -1,
  total: number,
  deadFavorites: ReadonlySet<number>,
): number {
  if (total <= 0) return 0;
  let idx = fromFav;
  for (let step = 0; step < total; step += 1) {
    idx = delta === 1 ? nextFavoriteIndex(idx, total) : prevFavoriteIndex(idx, total);
    if (!deadFavorites.has(idx)) return idx;
  }
  return Math.max(0, fromFav < 0 ? 0 : fromFav);
}

/** Page index (≥1) for a favorite array index. */
export function pageIndexForFavorite(favoriteIndex: number): number {
  return favoriteIndex + 1;
}

/** Starting favorite index when stepping from a cable page. */
export function stepFromFavoriteIndex(pageIndex: number, delta: 1 | -1, total: number): number {
  if (isQuadPage(pageIndex)) return delta === 1 ? -1 : total;
  return singleFavoriteIndex(pageIndex);
}

/** Target favorite after CH ▲/▼, skipping dead streams. */
export function targetFavoriteAfterStep(
  pageIndex: number,
  delta: 1 | -1,
  total: number,
  deadFavorites: ReadonlySet<number>,
): number {
  const from = stepFromFavoriteIndex(pageIndex, delta, total);
  return nextWorkingFavoriteIndex(from, delta, total, deadFavorites);
}

/** Favorite index for the page you land on after CH ▲ from `pageIndex`. */
export function preloadFavoriteIndexForPage(pageIndex: number, total: number): number {
  if (total <= 0) return 0;
  const nextPage = nextCablePageIndex(pageIndex, 1, total);
  if (isQuadPage(nextPage)) return quadPreloadFavoriteIndex(initialRotationCursor(total), total);
  return singleFavoriteIndex(nextPage);
}
