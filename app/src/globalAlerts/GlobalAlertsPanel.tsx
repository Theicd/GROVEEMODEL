import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { enrichStormBriefing, type EnrichedStormBriefing } from "./enrichStormBriefing";
import { enrichNeoBriefing, type EnrichedNeoBriefing } from "./enrichNeoBriefing";
import { EventFocusDock } from "./EventFocusDock";
import { fetchNeoHorizonsTrack } from "./fetchNeoHorizonsTrack";
import { fetchStormBriefing } from "./fetchStormBriefing";
import { HurricaneBriefingCard } from "./HurricaneBriefingCard";
import { hurricaneColorCss } from "./hurricaneIntensity";
import { initGlobeScene, type GlobeSceneHandle } from "./globeScene";
import { getEventSeverity, SeverityMeter } from "./SeverityMeter";
import { EVENT_TYPE_LABELS, type GlobeAlertEvent } from "./types";
import { filterNeoAlerts } from "./alertFilters";
import { formatNeoEta } from "./neoEta";
import { liveNeoMetrics } from "./neoLiveMetrics";
import { buildSyntheticNeoTrack } from "./syntheticNeoTrack";
import { timeAgo, useGlobalAlertEvents } from "./useGlobalAlertEvents";
import "./globalAlerts.css";

type Props = {
  onClose: () => void;
};

function eventAccentColor(ev: GlobeAlertEvent): string {
  if (ev.type === "hurricane") return hurricaneColorCss(ev.category, ev.severityText);
  return EVENT_TYPE_LABELS[ev.type].color;
}

function eventDetail(ev: GlobeAlertEvent): string {
  const ti = EVENT_TYPE_LABELS[ev.type];
  if (ev.type === "earthquake" && ev.magnitude != null) {
    return `M${ev.magnitude.toFixed(1)}`;
  }
  if (ev.type === "hurricane" && ev.category != null) {
    return `קט ${ev.category}`;
  }
  if (ev.type === "neo") {
    return `${(ev.distLd ?? 0).toFixed(1)} LD`;
  }
  if (ev.alertLevel) {
    return ev.alertLevel;
  }
  return ti.label;
}

function EventFocusStrip({
  ev,
  onReturn,
}: {
  ev: GlobeAlertEvent;
  onReturn: () => void;
}) {
  const ti = EVENT_TYPE_LABELS[ev.type];
  const accent = eventAccentColor(ev);
  return (
    <div className="global-alerts-focus-strip" style={{ borderColor: accent }}>
      <div className="global-alerts-focus-strip__row">
        <span className="global-alerts-focus-strip__badge" style={{ color: accent, borderColor: accent }}>
          {ti.label}
        </span>
        <span className="global-alerts-focus-strip__detail">{eventDetail(ev)}</span>
        <button type="button" className="global-alerts-focus-strip__back" onClick={onReturn}>
          חזרה
        </button>
      </div>
      <p className="global-alerts-focus-strip__loc">{ev.location}</p>
      {ev.severityText ? (
        <p className="global-alerts-focus-strip__meta">{ev.severityText}</p>
      ) : null}
      <p className="global-alerts-focus-strip__meta">
        {ev.magnitude != null ? (
          <>
            מגניטודה <strong style={{ color: accent }}>M{ev.magnitude.toFixed(1)}</strong>
            {ev.depth != null ? ` · עומק ${ev.depth.toFixed(1)} ק"מ` : ""}
            {" · "}
          </>
        ) : null}
        {ev.category != null ? (
          <>
            קטגוריה <strong style={{ color: accent }}>{ev.category}</strong>
            {" · "}
          </>
        ) : null}
        {timeAgo(ev.time)} · {ev.source.toUpperCase()}
      </p>
      <SeverityMeter ev={ev} color={accent} compact />
    </div>
  );
}

export function GlobalAlertsPanel({ onClose }: Props) {
  const sceneRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<GlobeSceneHandle | null>(null);
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [sceneReady, setSceneReady] = useState(false);
  const [focused, setFocused] = useState<GlobeAlertEvent | null>(null);
  const [stormEnriched, setStormEnriched] = useState<EnrichedStormBriefing | null>(null);
  const [trackLoading, setTrackLoading] = useState(false);
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 768px)").matches,
  );
  const [neoTracks, setNeoTracks] = useState<Record<string, import("./neoTrack").NeoOrbitTrack>>({});
  const [neoEnriched, setNeoEnriched] = useState<EnrichedNeoBriefing | null>(null);
  const [neoBriefCache, setNeoBriefCache] = useState<Record<string, EnrichedNeoBriefing>>({});
  const [liveTick, setLiveTick] = useState(0);
  const { events, loading, lastUpdate, eqCount, gdacsCount } = useGlobalAlertEvents(true);

  useEffect(() => {
    const id = window.setInterval(() => setLiveTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 768px)");
    const sync = () => setIsMobile(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  const showHeaderFocus = isMobile && !!focused && panelCollapsed;
  const showFocusDock = !isMobile && !!focused && panelCollapsed;

  const loadStormBriefing = useCallback(async (ev: GlobeAlertEvent) => {
    if (ev.type !== "hurricane" || ev.gdacsEventId == null || ev.gdacsEpisodeId == null) {
      setStormEnriched(null);
      globeRef.current?.clearStormTrack();
      return;
    }
    globeRef.current?.clearNeoTrack();
    setTrackLoading(true);
    const briefing = await fetchStormBriefing(ev.gdacsEventId, ev.gdacsEpisodeId);
    if (!briefing) {
      setTrackLoading(false);
      setStormEnriched(null);
      return;
    }
    globeRef.current?.showStormTrack(briefing.track);
    globeRef.current?.focusEvent(ev);
    const enriched = await enrichStormBriefing(briefing, ev);
    setStormEnriched(enriched);
    setTrackLoading(false);
  }, []);

  const loadNeoTrack = useCallback(async (ev: GlobeAlertEvent) => {
    if (ev.type !== "neo" || !ev.designation) {
      globeRef.current?.clearNeoTrack();
      return;
    }
    globeRef.current?.clearStormTrack();
    setStormEnriched(null);
    setTrackLoading(true);
    globeRef.current?.focusEvent(ev);

    const briefPromise = enrichNeoBriefing(ev);

    let track = await fetchNeoHorizonsTrack(ev.designation, ev.approachTime ?? ev.time, 14).catch(
      () => null,
    );
    if (!track || track.points.length < 2) {
      track = buildSyntheticNeoTrack(ev);
    }
    const brief = await briefPromise;
    setNeoEnriched(brief);
    setNeoBriefCache((prev) => ({ ...prev, [ev.id]: brief }));
    setTrackLoading(false);
    setNeoTracks((prev) => ({ ...prev, [ev.id]: track! }));
    globeRef.current?.showNeoTrack(track!, ev.diameterKm);
    globeRef.current?.focusNeoEarthFrame(track!);
  }, []);

  const loadEventFocus = useCallback(
    async (ev: GlobeAlertEvent) => {
      if (ev.type === "hurricane") {
        await loadStormBriefing(ev);
        return;
      }
      if (ev.type === "neo") {
        await loadNeoTrack(ev);
        return;
      }
      setStormEnriched(null);
      setTrackLoading(false);
      globeRef.current?.clearStormTrack();
      globeRef.current?.clearNeoTrack();
    },
    [loadStormBriefing, loadNeoTrack],
  );

  const handleCardClick = useCallback(
    (ev: GlobeAlertEvent) => {
      setPanelCollapsed(true);
      setFocused(ev);
      globeRef.current?.focusEvent(ev);
      void loadEventFocus(ev);
    },
    [loadEventFocus],
  );

  const handleReturn = useCallback(() => {
    setFocused(null);
    setStormEnriched(null);
    setNeoEnriched(null);
    setTrackLoading(false);
    globeRef.current?.returnToNormal();
  }, []);

  useEffect(() => {
    if (!sceneRef.current) return;
    globeRef.current = initGlobeScene(sceneRef.current, {
      onLoaded: () => setSceneReady(true),
      onFocus: (ev) => setFocused(ev),
      onEventPick: (ev) => handleCardClick(ev),
    });
    return () => {
      globeRef.current?.dispose();
      globeRef.current = null;
    };
  }, [handleCardClick]);

  useEffect(() => {
    if (!globeRef.current || !events.length) return;
    globeRef.current.syncEvents(events);
  }, [events, sceneReady]);

  const neoEvents = useMemo(() => filterNeoAlerts(events.filter((e) => e.type === "neo")), [events]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (const ev of neoEvents) {
        const brief = await enrichNeoBriefing(ev);
        if (cancelled) return;
        setNeoBriefCache((prev) => (prev[ev.id] ? prev : { ...prev, [ev.id]: brief }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [neoEvents]);

  useEffect(() => {
    if (!focused || (focused.type !== "hurricane" && focused.type !== "neo")) return;
    const id = window.setInterval(() => {
      void loadEventFocus(focused);
    }, 60_000);
    return () => window.clearInterval(id);
  }, [focused, loadEventFocus]);

  const typeCounts = useMemo(() => {
    const counts: Partial<Record<GlobeAlertEvent["type"], number>> = {};
    for (const ev of events) {
      if (ev.type === "neo") continue;
      counts[ev.type] = (counts[ev.type] ?? 0) + 1;
    }
    return counts;
  }, [events]);

  const neoCountFiltered = neoEvents.length;

  const renderEventCard = (ev: GlobeAlertEvent) => {
    const ti = EVENT_TYPE_LABELS[ev.type];
    const accent = eventAccentColor(ev);
    const isActive = focused?.id === ev.id;
    const sev = getEventSeverity(ev);
    const track = ev.type === "neo" ? neoTracks[ev.id] : undefined;
    void liveTick;
    const live = ev.type === "neo" ? liveNeoMetrics(ev, track) : null;
    const brief = ev.type === "neo" ? neoBriefCache[ev.id] : undefined;
    return (
      <div
        key={ev.id}
        className={`global-alerts-card is-sev-${sev.tier}${isActive ? " is-active" : ""}`}
        style={{ borderRightColor: accent, "--sev-color": accent } as CSSProperties}
        onClick={() => handleCardClick(ev)}
        onKeyDown={(e) => e.key === "Enter" && handleCardClick(ev)}
        role="button"
        tabIndex={0}
      >
        <div className="global-alerts-card__top">
          <div className="global-alerts-card__type" style={{ color: accent }}>
            {ti.label}
          </div>
          <span className="global-alerts-card__mag" style={{ color: accent }}>
            {ev.type === "neo" && live ? `${live.distLd.toFixed(1)} LD` : eventDetail(ev)}
          </span>
        </div>
        <div className="global-alerts-card__loc">{ev.location}</div>
        {ev.type === "neo" && brief ? (
          <div className="global-alerts-card__neo-brief">
            {brief.impactRegion.split(" · ").pop()} · סכנה {brief.riskLabel}
          </div>
        ) : null}
        <div className="global-alerts-card__foot">
          <SeverityMeter ev={ev} color={accent} compact />
          <span className="global-alerts-card__time">
            {ev.type === "neo" && live
              ? `${live.speedKmS.toFixed(1)} km/s · ${formatNeoEta(ev.approachTime ?? ev.time)}`
              : timeAgo(ev.time)}
          </span>
        </div>
      </div>
    );
  };

  const updateLabel = lastUpdate
    ? new Date(lastUpdate).toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" })
    : "--:--";

  const showLoading = loading && !sceneReady;

  return (
    <div
      className={`global-alerts-root${showFocusDock ? " global-alerts-root--focus-dock" : ""}`}
      role="dialog"
      aria-label="מערכת התרעות גלובלית"
    >
      <div className={`global-alerts-loading${showLoading ? "" : " is-hidden"}`}>
        <div className="global-alerts-loader" />
        <div className="global-alerts-loading-text">טוען מערכת התרעות גלובלית...</div>
      </div>

      <header className={`global-alerts-header${showHeaderFocus ? " global-alerts-header--focused" : ""}`}>
        <button type="button" className="global-alerts-close" onClick={onClose} aria-label="סגור">
          ✕
        </button>
        <div className="global-alerts-header__body">
          <h1 className="global-alerts-header__title">Global Alert System</h1>
          {showHeaderFocus && focused ? (
            focused.type === "hurricane" ? (
              <HurricaneBriefingCard
                ev={focused}
                enriched={stormEnriched}
                loading={trackLoading}
                onReturn={handleReturn}
              />
            ) : focused.type === "neo" ? (
              <div className="global-alerts-header-dock">
                <EventFocusDock
                  ev={focused}
                  enriched={stormEnriched}
                  loading={trackLoading}
                  neoTrack={neoTracks[focused.id] ?? null}
                  neoEnriched={neoEnriched}
                  onReturn={handleReturn}
                />
              </div>
            ) : (
              <EventFocusStrip ev={focused} onReturn={handleReturn} />
            )
          ) : (
            <p className="global-alerts-header__hint">
              <span className="global-alerts-hdot" style={{ background: "#3498db" }} /> סיבוב
              <span className="global-alerts-hdot" style={{ background: "#3498db" }} /> זום
              <span className="global-alerts-hdot" style={{ background: "#ffeb3b" }} /> לחץ על אירוע
            </p>
          )}
        </div>
      </header>

      <div className="global-alerts-legend" aria-label="מקרא אירועים על כדור הארץ">
        <span className="global-alerts-legend__title">מקרא:</span>
        {(
          [
            ["earthquake", "רעידה · 10 דק׳ אחרונות"],
            ["tsunami", "צונאמי · GDACS/USGS"],
            ["hurricane", "הוריקן / סופה (GDACS)"],
            ["fire", "שריפה (GDACS)"],
            ["flood", "שיטפון (GDACS)"],
            ["volcano", "הר געש (GDACS)"],
            ["neo", "אסטרואיד בדרך (CAD)"],
          ] as const
        ).map(([type, hint]) => (
          <span key={type} className="global-alerts-legend__item">
            <span className="global-alerts-legend__dot" style={{ background: EVENT_TYPE_LABELS[type].color }} />
            {hint}
          </span>
        ))}
        <span className="global-alerts-legend__item">
          <span className="global-alerts-legend__dot" style={{ background: "#3399ff" }} />
          גשם (Open-Meteo)
        </span>
        <span className="global-alerts-legend__item">
          <span className="global-alerts-legend__dot" style={{ background: "#ffee55" }} />
          רעמים
        </span>
        <span className="global-alerts-legend__item">
          <span className="global-alerts-legend__dot" style={{ background: "#ccc" }} />
          עננים
        </span>
      </div>

      <aside className={`global-alerts-panel${panelCollapsed ? " is-collapsed" : ""}`}>
        <div className="global-alerts-panel__header">
          <h2>התרעות בזמן אמת</h2>
          <div className="global-alerts-panel__counts">
            {eqCount > 0 ? (
              <span className="global-alerts-cbadge">
                <span className="global-alerts-cdot" style={{ background: "#FF4444" }} />
                {eqCount} רעידות · 10 דק׳
              </span>
            ) : null}
            {gdacsCount > 0 ? (
              <span className="global-alerts-cbadge">
                <span className="global-alerts-cdot" style={{ background: "#AA66FF" }} />
                {gdacsCount} אסונות פעילים
              </span>
            ) : null}
            {neoCountFiltered > 0 ? (
              <span className="global-alerts-cbadge global-alerts-cbadge--neo">
                <span className="global-alerts-cdot" style={{ background: EVENT_TYPE_LABELS.neo.color }} />
                {neoCountFiltered} אסטרואידים · 48ש׳
              </span>
            ) : null}
            {Object.entries(typeCounts).map(([t, n]) => {
              const type = t as GlobeAlertEvent["type"];
              if (type === "earthquake" || type === "disaster") return null;
              const ti = EVENT_TYPE_LABELS[type];
              return (
                <span key={type} className="global-alerts-cbadge">
                  <span className="global-alerts-cdot" style={{ background: ti.color }} />
                  {ti.label}: {n}
                </span>
              );
            })}
          </div>
          {(eqCount > 0 || gdacsCount > 0 || neoCountFiltered > 0) ? (
            <p className="global-alerts-panel__live-note">
              חי · USGS 10 דק׳ · GDACS פעיל · JPL · עודכן {updateLabel}
            </p>
          ) : null}
        </div>

        <div className="global-alerts-list">
          {!events.length && !loading ? (
            <p className="global-alerts-empty">
              אין נתוני אירועים כרגע — USGS/GDACS/JPL מתעדכנים ברקע.
            </p>
          ) : null}
          {events.slice(0, 50).map((ev) => renderEventCard(ev))}
        </div>

        <div className="global-alerts-panel__footer">
          <span>
            <span className="global-alerts-sdot" />
            חי
          </span>
          <span>{updateLabel}</span>
        </div>
      </aside>

      <button
        type="button"
        className={`global-alerts-toggle${panelCollapsed ? " is-shifted" : ""}`}
        onClick={() => setPanelCollapsed((v) => !v)}
        aria-label={panelCollapsed ? "פתח פאנל" : "סגור פאנל"}
      >
        {panelCollapsed ? "▶" : "◀"}
      </button>

      {showFocusDock && focused ? (
        <aside className="global-alerts-focus-dock" aria-label="פרטי אירוע">
          <EventFocusDock
            ev={focused}
            enriched={stormEnriched}
            loading={trackLoading}
            neoTrack={focused.type === "neo" ? neoTracks[focused.id] ?? null : null}
            neoEnriched={focused.type === "neo" ? neoEnriched : null}
            onReturn={handleReturn}
          />
        </aside>
      ) : null}

      <div ref={sceneRef} className="global-alerts-scene" />
    </div>
  );
}
