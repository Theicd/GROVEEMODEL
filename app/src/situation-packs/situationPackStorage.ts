/** Persist enabled/disabled pack overrides. */

import { DEFAULT_SITUATION_PACKS } from "./defaultPacks";
import type { SituationPack } from "./types";

const STORAGE_KEY = "grovee-situation-packs-v1";

export const loadSituationPacks = (): SituationPack[] => {
  if (typeof localStorage === "undefined") return DEFAULT_SITUATION_PACKS.map(clonePack);
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SITUATION_PACKS.map(clonePack);
    const overrides = JSON.parse(raw) as Record<string, { enabled?: boolean }>;
    return DEFAULT_SITUATION_PACKS.map((pack) => ({
      ...clonePack(pack),
      enabled: overrides[pack.id]?.enabled ?? pack.enabled,
    }));
  } catch {
    return DEFAULT_SITUATION_PACKS.map(clonePack);
  }
};

export const saveSituationPackOverrides = (packs: SituationPack[]): void => {
  if (typeof localStorage === "undefined") return;
  const overrides: Record<string, { enabled: boolean }> = {};
  for (const pack of packs) {
    const def = DEFAULT_SITUATION_PACKS.find((d) => d.id === pack.id);
    if (def && def.enabled !== pack.enabled) {
      overrides[pack.id] = { enabled: pack.enabled };
    }
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
};

const clonePack = (pack: SituationPack): SituationPack => ({
  ...pack,
  triggers: { ...pack.triggers },
  responses: [...pack.responses],
  sceneTags: pack.sceneTags ? [...pack.sceneTags] : undefined,
});
