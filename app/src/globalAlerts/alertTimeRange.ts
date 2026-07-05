import { filterSidebarAlerts } from "./alertFilters";
import type { GlobeAlertEvent } from "./types";

export type AlertTimeRange = "live" | "space";

export const ALERT_LIVE_RANGE_OPTION = { id: "live" as const, label: "30 דקות" };
export const ALERT_SPACE_RANGE_OPTION = { id: "space" as const, label: "התרעות חלל" };

export const ALERT_TIME_RANGE_OPTIONS: { id: AlertTimeRange; label: string }[] = [
  ALERT_LIVE_RANGE_OPTION,
  ALERT_SPACE_RANGE_OPTION,
];

/** Apply tab filter on fetched events. */
export function filterEventsForRange(
  events: GlobeAlertEvent[],
  range: AlertTimeRange,
  _now = Date.now(),
): GlobeAlertEvent[] {
  if (range === "space") {
    return events
      .filter((e) => e.type === "neo" || e.type === "fireball" || e.showcaseNeo)
      .sort((a, b) => {
        if (a.showcaseNeo !== b.showcaseNeo) return a.showcaseNeo ? 1 : -1;
        const da = a.approachTime ?? a.time;
        const db = b.approachTime ?? b.time;
        if (da !== db) return da - db;
        return (a.distLd ?? 99) - (b.distLd ?? 99);
      });
  }
  return filterSidebarAlerts(events);
}

export function rangeLabel(range: AlertTimeRange): string {
  return ALERT_TIME_RANGE_OPTIONS.find((o) => o.id === range)?.label ?? range;
}
