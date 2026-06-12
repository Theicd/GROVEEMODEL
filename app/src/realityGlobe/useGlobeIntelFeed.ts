import { useCallback, useEffect, useRef, useState } from "react";
import type { GlobeGaugeCard } from "./GlobeGaugeStrip";
import type { GlobeLayersState } from "./bridge";
import {
  fetchGlobeIntelSnapshot,
  type GlobeIntelSnapshot,
  type IntelFlashAlert,
  type IntelHeadline,
  type IntelTickerItem,
} from "./intelFeed";
import {
  isRealtimeFlash,
  loadSeenFlashes,
  loadTickerHistory,
  markFlashSeen,
  saveTickerHistory,
} from "./globeIntelStore";
import { mergeTickerHistory } from "./tickerUtils";

export function useGlobeIntelFeed(active: boolean) {
  const [snapshot, setSnapshot] = useState<GlobeIntelSnapshot>({
    tickers: [],
    headlines: [],
    flash: null,
  });
  const [timeline, setTimeline] = useState<IntelTickerItem[]>(() => [...loadTickerHistory().values()]);
  const [gauges, setGauges] = useState<GlobeGaugeCard[]>([]);
  const [updatedAt, setUpdatedAt] = useState("");
  const [activeFlash, setActiveFlash] = useState<IntelFlashAlert | null>(null);
  const seenFlashRef = useRef(loadSeenFlashes());
  const historyRef = useRef(loadTickerHistory());
  const panelOpenTsRef = useRef(Date.now());

  const tryShowFlash = useCallback((flash: IntelFlashAlert | null, eventTs?: number) => {
    if (!flash || seenFlashRef.current.has(flash.id)) return;
    if (!isRealtimeFlash(flash, eventTs)) return;
    // Don't popup stale events right when panel opens (only truly new after open)
    if (eventTs && eventTs < panelOpenTsRef.current - 60_000) return;
    seenFlashRef.current.add(flash.id);
    markFlashSeen(flash.id);
    setActiveFlash(flash);
  }, []);

  const ingestTickers = useCallback((incoming: IntelTickerItem[]) => {
    if (!incoming.length) return;
    const merged = mergeTickerHistory(incoming, historyRef.current);
    saveTickerHistory(historyRef.current);
    setTimeline(merged);
    setSnapshot((prev) => ({
      tickers: merged,
      headlines: prev.headlines,
      flash: prev.flash,
    }));
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchGlobeIntelSnapshot();
      if (next.tickers.length) ingestTickers(next.tickers);
      setSnapshot(() => ({
        tickers: historyRef.current.size ? [...historyRef.current.values()] : next.tickers,
        headlines: next.headlines,
        flash: null,
      }));
    } catch {
      /* keep last snapshot */
    }
  }, [ingestTickers]);

  useEffect(() => {
    if (!active) return;
    panelOpenTsRef.current = Date.now();
    void refresh();
  }, [active, refresh]);

  useEffect(() => {
    if (!active) return;
    const onMsg = (e: MessageEvent) => {
      if (e.data?.source !== "reality-core") return;

      if (e.data.type === "intel" && e.data.payload) {
        const p = e.data.payload as {
          tickers?: IntelTickerItem[];
          headlines?: IntelHeadline[];
          gauges?: GlobeGaugeCard[];
          updatedAt?: string;
          flash?: IntelFlashAlert | null;
          flashTs?: number;
        };
        const headlineItems: IntelTickerItem[] = (p.headlines ?? []).map((h) => ({
          id: `hl-${h.id}`,
          severity: h.severity,
          tag: h.severity >= 5 ? "BREAKING" : "כותרת",
          text: h.text,
          time: "",
          ts: Date.now(),
          category: "HEADLINE",
        }));
        ingestTickers([...(p.tickers ?? []), ...headlineItems]);
        if (Array.isArray(p.gauges)) setGauges(p.gauges);
        if (p.updatedAt) setUpdatedAt(p.updatedAt);
        setSnapshot((prev) => ({
          ...prev,
          headlines: p.headlines ?? prev.headlines,
        }));
      }

      if (e.data.type === "live_alert" && e.data.payload) {
        const p = e.data.payload as IntelFlashAlert & { ts?: number };
        tryShowFlash(
          {
            id: p.id,
            severity: p.severity,
            title: p.title,
            body: p.body,
            category: p.category,
            lat: p.lat,
            lon: p.lon,
          },
          p.ts,
        );
        ingestTickers([
          {
            id: `live-${p.id}`,
            severity: p.severity,
            tag: p.category || "LIVE",
            text: p.body,
            time: "",
            ts: p.ts ?? Date.now(),
            icon: p.category === "ISRAEL" ? "🚨" : p.category === "SEISMIC" ? "🌍" : "🔴",
            category: p.category,
            lat: p.lat,
            lon: p.lon,
          },
        ]);
      }
    };
    window.addEventListener("message", onMsg);
    const poll = window.setInterval(() => void refresh(), 60_000);
    return () => {
      window.removeEventListener("message", onMsg);
      window.clearInterval(poll);
    };
  }, [active, refresh, ingestTickers, tryShowFlash]);

  useEffect(() => {
    if (!activeFlash) return;
    const t = window.setTimeout(() => setActiveFlash(null), 9000);
    return () => window.clearTimeout(t);
  }, [activeFlash]);

  const dismissFlash = useCallback(() => {
    if (activeFlash) markFlashSeen(activeFlash.id);
    setActiveFlash(null);
  }, [activeFlash]);

  return {
    snapshot,
    timeline,
    gauges,
    updatedAt,
    activeFlash,
    dismissFlash,
  };
}

export function useGlobeLayersFromMessage(
  onLayers: (layers: GlobeLayersState) => void,
): void {
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.data?.source !== "reality-core" || e.data.type !== "layers") return;
      const payload = e.data.payload as GlobeLayersState;
      if (payload && typeof payload === "object") onLayers(payload);
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [onLayers]);
}
