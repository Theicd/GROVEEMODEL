import { useCallback, useEffect, useState } from "react";
import { subscribeLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import { filterSidebarAlerts, filterSpacePanelNeos } from "./alertFilters";
import { fetchGlobalAlertEventsForRange } from "./fetchGlobalAlertEvents";
import { rawSnapshotToGlobeEvents } from "./mapHitsToGlobeEvents";
import type { AlertTimeRange } from "./alertTimeRange";
import type { GlobeAlertEvent } from "./types";

const REFRESH_MS = 60_000;

export function useGlobalAlertEvents(active: boolean, timeRange: AlertTimeRange = "live") {
  const [events, setEvents] = useState<GlobeAlertEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [eqCount, setEqCount] = useState(0);
  const [gdacsCount, setGdacsCount] = useState(0);
  const [neoCount, setNeoCount] = useState(0);

  const applyEvents = useCallback(
    (next: GlobeAlertEvent[]) => {
      setEvents(next);
      setEqCount(next.filter((e) => e.source === "usgs").length);
      setGdacsCount(next.filter((e) => e.source === "gdacs").length);
      setNeoCount(
        timeRange === "space" ? filterSpacePanelNeos(next).length : 0,
      );
      setLastUpdate(Date.now());
      setLoading(false);
    },
    [timeRange],
  );

  const refresh = useCallback(async () => {
    try {
      const next = await fetchGlobalAlertEventsForRange(timeRange);
      applyEvents(next);
    } catch {
      setLoading(false);
    }
  }, [applyEvents, timeRange]);

  useEffect(() => {
    if (!active) return;

    setEvents([]);
    setLoading(true);
    void refresh();

    const unsub =
      timeRange === "live"
        ? subscribeLiveWorldSnapshot((snap) => {
            if (!snap) return;
            const fromSnap = rawSnapshotToGlobeEvents(snap);
            if (!fromSnap.length) return;
            setEvents((prev) => {
              const filtered = filterSidebarAlerts(fromSnap);
              setEqCount(filtered.filter((e) => e.source === "usgs").length);
              setGdacsCount(filtered.filter((e) => e.source === "gdacs").length);
              setNeoCount(0);
              setLastUpdate(Date.now());
              return filtered.length ? filtered : prev;
            });
          })
        : undefined;

    const id = window.setInterval(() => void refresh(), REFRESH_MS);
    return () => {
      unsub?.();
      window.clearInterval(id);
    };
  }, [active, refresh, timeRange]);

  return { events, loading, lastUpdate, eqCount, gdacsCount, neoCount, refresh };
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `לפני ${s} שניות`;
  const m = Math.floor(s / 60);
  if (m < 60) return `לפני ${m} דקות`;
  const h = Math.floor(m / 60);
  if (h < 24) return `לפני ${h} שעות`;
  return `לפני ${Math.floor(h / 24)} ימים`;
}

export { timeAgo };
