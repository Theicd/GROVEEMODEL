/** Pick response variant — mood, repetition, anti-repeat. */

import type { CharacterMood } from "../characterBrain";
import type { SituationPack } from "./types";

export type VariationState = {
  packFireCounts: Map<string, number>;
  recentTexts: string[];
};

export const createVariationState = (): VariationState => ({
  packFireCounts: new Map(),
  recentTexts: [],
});

const moodOffset = (mood: CharacterMood, len: number): number => {
  switch (mood) {
    case "excited":
      return Math.floor(len * 0.75);
    case "curious":
      return Math.floor(len * 0.35);
    case "observing":
    default:
      return 0;
  }
};

export const pickResponseVariant = (
  pack: SituationPack,
  state: VariationState,
  mood: CharacterMood,
): string => {
  const responses = pack.responses.filter(Boolean);
  if (!responses.length) return pack.interpretation;

  const unused = responses.filter((r) => !state.recentTexts.includes(r));
  const pool = unused.length ? unused : responses;

  const fireCount = state.packFireCounts.get(pack.id) ?? 0;
  const base = moodOffset(mood, pool.length);
  const idx = (base + fireCount) % pool.length;
  return pool[idx] ?? pool[0];
};

export const noteVariantUsed = (
  state: VariationState,
  packId: string,
  text: string,
  maxRecent = 12,
): void => {
  state.packFireCounts.set(packId, (state.packFireCounts.get(packId) ?? 0) + 1);
  state.recentTexts = [text, ...state.recentTexts.filter((t) => t !== text)].slice(0, maxRecent);
};

export const resetVariationState = (state: VariationState): void => {
  state.packFireCounts.clear();
  state.recentTexts = [];
};
