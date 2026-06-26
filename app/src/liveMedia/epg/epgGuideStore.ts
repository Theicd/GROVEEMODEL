import type { UnifiedSearchHit } from "../../searchResults/types";
import { fetchEpgSchedule, hitToEpgLookup, warmMjhEpgCaches } from "./epgService";
import { fetchMjhXmltv } from "./mjhSources";
import type { EpgSchedule } from "./types";

export type EpgGuideEntry = {
  hit: UnifiedSearchHit;
  schedule: EpgSchedule | null;
};

let guideCache: EpgGuideEntry[] | null = null;
let guidePending: Promise<EpgGuideEntry[]> | null = null;
let guideKey = "";

function guideCacheKey(hits: UnifiedSearchHit[]): string {
  return `v4|${hits.map((h) => `${h.id}:${h.title}`).join("|")}`;
}

function sortEntries(entries: EpgGuideEntry[]): EpgGuideEntry[] {
  return [...entries].sort((a, b) => {
    const aOk = (a.schedule?.programs.length ?? 0) > 0 ? 0 : 1;
    const bOk = (b.schedule?.programs.length ?? 0) > 0 ? 0 : 1;
    if (aOk !== bOk) return aOk - bOk;
    return a.hit.title.localeCompare(b.hit.title, undefined, { sensitivity: "base" });
  });
}

export function resetEpgGuideCacheForTests(): void {
  guideCache = null;
  guidePending = null;
  guideKey = "";
}

export async function verifyEpgFeedsLoaded(): Promise<boolean> {
  const xml = await fetchMjhXmltv("https://i.mjh.nz/all/epg.xml.gz");
  return xml != null && xml.length > 1000;
}

/** Load EPG schedules for every favorite — progressive updates for the grid UI. */
export async function loadEpgGuide(
  hits: UnifiedSearchHit[],
  onProgress?: (entries: EpgGuideEntry[], loaded: number, total: number) => void,
): Promise<EpgGuideEntry[]> {
  const key = guideCacheKey(hits);
  if (guideCache && guideKey === key) return guideCache;
  if (guidePending && guideKey === key) return guidePending;

  guideKey = key;
  guidePending = (async () => {
    await warmMjhEpgCaches();

    const tvHits = hits.filter((h) => h.kind === "livetv" || h.kind === "youtube");
    const entries: EpgGuideEntry[] = [];
    const batchSize = 3;

    for (let i = 0; i < tvHits.length; i += batchSize) {
      const batch = tvHits.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(async (hit) => {
          const lookup = hitToEpgLookup(hit);
          if (!lookup) return { hit, schedule: null as const };
          const schedule = await fetchEpgSchedule(lookup, { guide: true });
          return { hit, schedule };
        }),
      );
      entries.push(...batchResults);
      onProgress?.(sortEntries(entries), Math.min(i + batchSize, tvHits.length), tvHits.length);
    }

    const sorted = sortEntries(entries);
    guideCache = sorted;
    guidePending = null;
    return sorted;
  })();

  return guidePending;
}

export function getEpgGuideSnapshot(): EpgGuideEntry[] | null {
  return guideCache;
}

export function countGuideEntriesWithData(entries: EpgGuideEntry[]): number {
  return entries.filter((e) => (e.schedule?.programs.length ?? 0) > 0).length;
}
