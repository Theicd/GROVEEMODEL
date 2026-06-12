import type { IntelFlashAlert, IntelTickerItem } from "./intelFeed";

const SEEN_FLASH_KEY = "grovee-globe-seen-flashes";
const TICKER_HISTORY_KEY = "grovee-globe-ticker-history";
const MS_24H = 24 * 60 * 60 * 1000;
/** Flash popup only for events within this window */
export const FLASH_MAX_AGE_MS = 20 * 60 * 1000;

export function loadSeenFlashes(): Set<string> {
  try {
    const raw = sessionStorage.getItem(SEEN_FLASH_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw) as string[];
    return new Set(Array.isArray(arr) ? arr : []);
  } catch {
    return new Set();
  }
}

export function markFlashSeen(id: string): void {
  const set = loadSeenFlashes();
  set.add(id);
  const arr = [...set].slice(-80);
  try {
    sessionStorage.setItem(SEEN_FLASH_KEY, JSON.stringify(arr));
  } catch {
    /* ignore */
  }
}

export function loadTickerHistory(): Map<string, IntelTickerItem> {
  try {
    const raw = sessionStorage.getItem(TICKER_HISTORY_KEY);
    if (!raw) return new Map();
    const arr = JSON.parse(raw) as IntelTickerItem[];
    const cutoff = Date.now() - MS_24H;
    const map = new Map<string, IntelTickerItem>();
    for (const item of arr) {
      if ((item.ts ?? 0) >= cutoff) map.set(item.id, item);
    }
    return map;
  } catch {
    return new Map();
  }
}

export function saveTickerHistory(store: Map<string, IntelTickerItem>): void {
  try {
    const arr = [...store.values()].slice(0, 120);
    sessionStorage.setItem(TICKER_HISTORY_KEY, JSON.stringify(arr));
  } catch {
    /* ignore */
  }
}

export function isRealtimeFlash(flash: IntelFlashAlert, eventTs?: number): boolean {
  if (flash.category === "ISRAEL") return true;
  if (eventTs == null) return false;
  return Date.now() - eventTs < FLASH_MAX_AGE_MS;
}
