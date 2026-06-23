import { describe, expect, it } from "vitest";
import type { UnifiedSearchHit } from "../searchResults/types";
import {
  advanceQuadRotation,
  cableOsdRangeLabel,
  favoriteForPage,
  initialQuadSlots,
  initialRotationCursor,
  isQuadPage,
  maxCablePageIndex,
  nextCablePageIndex,
  nextFavoriteIndex,
  nextWorkingFavoriteIndex,
  pageIndexForFavorite,
  pickCableQuadFromSlots,
  prevFavoriteIndex,
  quadOsdChannelRange,
  singleFavoriteIndex,
  targetFavoriteAfterStep,
} from "./cableTunerUtils";

function hit(id: string): UnifiedSearchHit {
  return { id, title: id, url: `https://example.com/${id}`, mediaPlayUrl: `https://example.com/${id}.m3u8` };
}

describe("cableTunerUtils", () => {
  it("page 0 is quad only", () => {
    expect(isQuadPage(0)).toBe(true);
    expect(isQuadPage(1)).toBe(false);
    expect(initialQuadSlots(66)).toEqual([0, 1, 2, 3]);
    expect(initialRotationCursor(66)).toBe(4);
  });

  it("quad rotation advances one slot at a time through all favorites", () => {
    let slots = initialQuadSlots(6);
    let cursor = initialRotationCursor(6);
    ({ slots, cursor } = advanceQuadRotation(slots, 0, cursor, 6));
    expect(slots).toEqual([4, 1, 2, 3]);
    expect(cursor).toBe(5);
    ({ slots, cursor } = advanceQuadRotation(slots, 1, cursor, 6));
    expect(slots).toEqual([4, 5, 2, 3]);
    expect(cursor).toBe(0);
  });

  it("pages after 0 are single channel in favorites order", () => {
    const hits = ["a", "b", "c", "d", "e", "f"].map(hit);
    expect(favoriteForPage(hits, 1)?.id).toBe("a");
    expect(favoriteForPage(hits, 5)?.id).toBe("e");
    expect(singleFavoriteIndex(3)).toBe(2);
    expect(cableOsdRangeLabel(3, 6)).toEqual({ from: 3, to: 3 });
  });

  it("quad osd reflects current slot indices not fixed 1-4", () => {
    expect(quadOsdChannelRange([8, 9, 10, 11])).toEqual({ from: 9, to: 12 });
    expect(cableOsdRangeLabel(0, 66, [8, 9, 10, 11])).toEqual({ from: 9, to: 12 });
  });

  it("advances one page at a time and wraps quad ↔ singles", () => {
    expect(maxCablePageIndex(66)).toBe(66);
    expect(nextCablePageIndex(0, 1, 66)).toBe(1);
    expect(nextCablePageIndex(1, 1, 66)).toBe(2);
    expect(nextCablePageIndex(66, 1, 66)).toBe(0);
    expect(nextCablePageIndex(0, -1, 66)).toBe(66);
  });

  it("maps quad slots to hits", () => {
    const hits = ["a", "b", "c", "d", "e"].map(hit);
    expect(pickCableQuadFromSlots(hits, [2, 3, 4, 0]).map((h) => h?.id)).toEqual(["c", "d", "e", "a"]);
  });

  it("skips dead favorites when stepping", () => {
    const dead = new Set([1, 2]);
    expect(nextWorkingFavoriteIndex(0, 1, 6, dead)).toBe(3);
    expect(nextWorkingFavoriteIndex(3, 1, 6, dead)).toBe(4);
    expect(targetFavoriteAfterStep(1, 1, 6, dead)).toBe(3);
    expect(targetFavoriteAfterStep(0, -1, 6, dead)).toBe(5);
    expect(pageIndexForFavorite(3)).toBe(4);
    expect(prevFavoriteIndex(0, 6)).toBe(5);
    expect(nextFavoriteIndex(-1, 6)).toBe(0);
  });
});
