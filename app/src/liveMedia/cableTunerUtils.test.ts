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
  pickNextQuadIndex,
  prevFavoriteIndex,
  quadOsdChannelRange,
  singleFavoriteIndex,
  spreadQuadSlots,
  targetFavoriteAfterStep,
} from "./cableTunerUtils";

function hit(id: string): UnifiedSearchHit {
  return { id, title: id, url: `https://example.com/${id}`, mediaPlayUrl: `https://example.com/${id}.m3u8` };
}

describe("cableTunerUtils", () => {
  it("page 0 is quad with spread slots, not 1–4 sequential", () => {
    expect(isQuadPage(0)).toBe(true);
    expect(isQuadPage(1)).toBe(false);
    expect(initialQuadSlots(66)).toEqual(spreadQuadSlots(66));
    expect(spreadQuadSlots(55)).toEqual([19, 25, 39, 52]);
    expect(spreadQuadSlots(66)).not.toEqual([0, 1, 2, 3]);
  });

  it("quad rotation avoids on-screen duplicates and spreads injections", () => {
    const total = 55;
    let slots = spreadQuadSlots(total);
    let cursor = initialRotationCursor(total, slots);
    expect(new Set(slots).size).toBe(4);

    for (let round = 0; round < 4; round += 1) {
      const before = [...slots];
      ({ slots, cursor } = advanceQuadRotation(slots, round, cursor, total));
      expect(new Set(slots).size).toBe(4);
      expect(slots[round]).not.toBe(before[round]);
      for (let i = 0; i < 4; i += 1) {
        if (i !== round) expect(slots[i]).toBe(before[i]);
      }
    }
  });

  it("pickNextQuadIndex never picks a channel already on another tile", () => {
    const slots = [19, 24, 39, 52];
    const next = pickNextQuadIndex(slots, 0, initialRotationCursor(55, slots), 55);
    expect([24, 39, 52]).not.toContain(next);
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
