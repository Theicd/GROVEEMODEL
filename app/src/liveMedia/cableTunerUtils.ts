import type { UnifiedSearchHit } from "../searchResults/types";

export type CableTile = UnifiedSearchHit | null;

const QUAD_TILES = 4;
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

/** Starting favorite indices for the quad grid (wraps). */
export function initialQuadSlots(total: number): number[] {
  if (total <= 0) return [0, 0, 0, 0];
  return Array.from({ length: QUAD_TILES }, (_, i) => i % total);
}

/** Index of the next favorite to inject into the quad rotation. */
export function initialRotationCursor(total: number): number {
  if (total <= 0) return 0;
  return QUAD_TILES % total;
}

/** Replace one quad tile with the next favorite in the lineup. */
export function advanceQuadRotation(
  slots: number[],
  slotToUpdate: number,
  cursor: number,
  total: number,
): { slots: number[]; cursor: number } {
  if (total <= 0) return { slots, cursor };
  const nextSlots = [...slots];
  nextSlots[slotToUpdate % QUAD_TILES] = cursor % total;
  return { slots: nextSlots, cursor: (cursor + 1) % total };
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
