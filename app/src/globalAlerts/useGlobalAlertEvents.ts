import { useCallback, useEffect, useState } from "react";
import { subscribeLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import { filterNeoAlerts, filterSidebarAlerts, sortAlertEvents } from "./alertFilters";
import { fetchGlobalAlertEvents } from "./fetchGlobalAlertEvents";
import { rawSnapshotToGlobeEvents } from "./mapHitsToGlobeEvents";
import type { GlobeAlertEvent } from "./types";

const REFRESH_MS = 60_000;

function mergeEarthWithNeo(earth: GlobeAlertEvent[], neo: GlobeAlertEvent[]): GlobeAlertEvent[] {
  const neoIds = new Set(neo.map((e) => e.id));
  const earthOnly = earth.filter((e) => e.type !== "neo" && !neoIds.has(e.id));
  return [...earthOnly, ...neo].sort(sortAlertEvents);
}

export function useGlobalAlertEvents(active: boolean) {
  const [events, setEvents] = useState<GlobeAlertEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [eqCount, setEqCount] = useState(0);
  const [gdacsCount, setGdacsCount] = useState(0);
  const [neoCount, setNeoCount] = useState(0);

  const applyEvents = useCallback((next: GlobeAlertEvent[]) => {
    const filtered = filterSidebarAlerts(next);
    setEvents(filtered);
    setEqCount(filtered.filter((e) => e.source === "usgs").length);
    setGdacsCount(filtered.filter((e) => e.source === "gdacs").length);
    setNeoCount(filterNeoAlerts(filtered).length);
    setLastUpdate(Date.now());
    setLoading(false);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchGlobalAlertEvents();
      applyEvents(next);
    } catch {
      setLoading(false);
    }
  }, [applyEvents]);

  useEffect(() => {
    if (!active) return;
    setLoading(true);
    void refresh();
    const unsub = subscribeLiveWorldSnapshot((snap) => {
      if (!snap) return;
      const fromSnap = rawSnapshotToGlobeEvents(snap);
      if (!fromSnap.length) return;
      setEvents((prev) => {
        const neos = prev.filter((e) => e.type === "neo");
        const merged = mergeEarthWithNeo(fromSnap, neos);
        const filtered = filterSidebarAlerts(merged);
        setEqCount(filtered.filter((e) => e.source === "usgs").length);
        setGdacsCount(filtered.filter((e) => e.source === "gdacs").length);
        setNeoCount(filterNeoAlerts(filtered).length);
        setLastUpdate(Date.now());
        return filtered;
      });
    });
    const id = window.setInterval(() => void refresh(), REFRESH_MS);
    return () => {
      unsub();
      window.clearInterval(id);
    };
  }, [active, refresh]);

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
