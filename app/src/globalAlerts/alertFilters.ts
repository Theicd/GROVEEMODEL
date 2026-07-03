import { getEventSeverity } from "./severityScore";
import type { GlobeAlertEvent } from "./types";
import { EARTH_LIVE_WINDOW_MS, EQ_LIVE_WINDOW_MS } from "./types";

export const NEO_ALERT_MAX_LD = 15;
export const NEO_ALERT_MAX_ETA_MS = 48 * 3_600_000;

/** @deprecated use EQ_LIVE_WINDOW_MS */
export const EQ_REALTIME_MAX_AGE_MS = EQ_LIVE_WINDOW_MS;

function tierAtLeast(tier: string, min: "moderate" | "high"): boolean {
  const order = { low: 0, moderate: 1, high: 2, critical: 3 };
  return (order[tier as keyof typeof order] ?? 0) >= order[min];
}

/** NEO: high/critical, ≤15 LD at CA, ETA within 48h. */
export function passesNeoAlertFilter(ev: GlobeAlertEvent): boolean {
  if (ev.type !== "neo") return false;
  const ld = ev.distLd ?? 99;
  if (ld > NEO_ALERT_MAX_LD) return false;
  const ca = ev.approachTime ?? ev.time;
  const eta = ca - Date.now();
  if (eta < -3_600_000) return false;
  if (eta > NEO_ALERT_MAX_ETA_MS) return false;
  return tierAtLeast(getEventSeverity(ev).tier, "high");
}

/** USGS earthquakes/tsunamis from the last 10 minutes — any magnitude. */
export function passesRealtimeEarthquakeFilter(ev: GlobeAlertEvent): boolean {
  if (ev.source !== "usgs") return true;
  if (ev.type !== "earthquake" && ev.type !== "tsunami") return true;
  return Date.now() - ev.time <= EQ_LIVE_WINDOW_MS;
}

const ONGOING_GDACS_TYPES = new Set<GlobeAlertEvent["type"]>([
  "hurricane",
  "fire",
  "flood",
  "volcano",
  "tsunami",
]);

/** GDACS: current orange/red ongoing hazards only; drop when episode ended. */
export function passesRealtimeGdacsFilter(ev: GlobeAlertEvent): boolean {
  if (ev.source !== "gdacs") return true;
  if (!/orange|red/i.test(ev.alertLevel ?? "")) return false;
  if (ev.gdacsIsCurrent === false) return false;
  const now = Date.now();
  if (ev.gdacsEndTime != null && ev.gdacsEndTime < now) return false;
  if (ONGOING_GDACS_TYPES.has(ev.type)) return true;
  const updated = ev.updatedTime ?? ev.time;
  return now - updated <= EARTH_LIVE_WINDOW_MS;
}

export function filterNeoAlerts(events: GlobeAlertEvent[]): GlobeAlertEvent[] {
  return events.filter(passesNeoAlertFilter);
}

export function filterSidebarAlerts(events: GlobeAlertEvent[]): GlobeAlertEvent[] {
  return events
    .filter((ev) => {
      if (ev.type === "neo") return passesNeoAlertFilter(ev);
      if (!passesRealtimeEarthquakeFilter(ev)) return false;
      if (!passesRealtimeGdacsFilter(ev)) return false;
      return true;
    })
    .sort(sortAlertEvents);
}

/** Earth events first (severity ↓), NEOs at the bottom (soonest ETA). */
export function sortAlertEvents(a: GlobeAlertEvent, b: GlobeAlertEvent): number {
  const aNeo = a.type === "neo";
  const bNeo = b.type === "neo";
  if (aNeo !== bNeo) return aNeo ? 1 : -1;

  if (aNeo && bNeo) {
    const da = a.approachTime ?? a.time;
    const db = b.approachTime ?? b.time;
    if (da !== db) return da - db;
    return (a.distLd ?? 99) - (b.distLd ?? 99);
  }

  const sa = getEventSeverity(a).score;
  const sb = getEventSeverity(b).score;
  if (sa !== sb) return sb - sa;
  return b.time - a.time;
}
