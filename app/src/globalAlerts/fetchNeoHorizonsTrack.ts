import { jplJson, parseJplUtcDate } from "./jplApi";
import { buildNeoTrackFromHorizons, type NeoOrbitTrack } from "./neoTrack";
import { parseHorizonsObserverRows } from "./parseHorizonsResult";

type HorizonsJson = {
  error?: string;
  result?: string;
};

const trackCache = new Map<string, { at: number; track: NeoOrbitTrack }>();
const CACHE_MS = 20 * 60_000;

function cacheKey(des: string, start: string, stop: string): string {
  return `${des}|${start}|${stop}`;
}

function formatHorizonsDate(ts: number): string {
  return new Date(ts).toISOString().slice(0, 10);
}

export async function fetchNeoHorizonsTrack(
  des: string,
  approachTime: number,
  windowDays = 5,
): Promise<NeoOrbitTrack | null> {
  const half = windowDays * 86400000;
  const start = formatHorizonsDate(approachTime - half);
  const stop = formatHorizonsDate(approachTime + half);
  const key = cacheKey(des, start, stop);
  const hit = trackCache.get(key);
  if (hit && Date.now() - hit.at < CACHE_MS) return hit.track;

  const command = `'DES=${des.replace(/'/g, "")};CAP'`;
  const url =
    `/horizons.api?format=json&EPHEM_TYPE=OBSERVER&COMMAND=${encodeURIComponent(command)}` +
    `&CENTER='500@399'&START_TIME='${start}'&STOP_TIME='${stop}'&STEP_SIZE='6h'` +
    `&QUANTITIES='1,20,23'&CAL_FORMAT=CAL&TIME_DIGITS=MINUTES&ANG_FORMAT=HMS`;

  const payload = await jplJson<HorizonsJson>(url, 35_000);
  if (payload.error || !payload.result) return null;

  const rows = parseHorizonsObserverRows(payload.result).map((r) => ({
    ...r,
    t: parseJplUtcDate(r.timeLabel),
  }));
  const track = buildNeoTrackFromHorizons(des, rows);
  if (track) trackCache.set(key, { at: Date.now(), track });
  return track;
}

export function clearNeoTrackCache(): void {
  trackCache.clear();
}
