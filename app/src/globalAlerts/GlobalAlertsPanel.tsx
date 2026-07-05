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
import { formatNeoCountdown } from "./neoEta";
import { liveNeoMetrics } from "./neoLiveMetrics";
import { buildSyntheticNeoTrack } from "./syntheticNeoTrack";
import { timeAgo, useGlobalAlertEvents } from "./useGlobalAlertEvents";
import {
  ALERT_TIME_RANGE_OPTIONS,
  rangeLabel,
  type AlertTimeRange,
} from "./alertTimeRange";
import { filterSpaceAlerts } from "./alertFilters";
import { formatAlertCardDisplay } from "./alertCardDisplay";
import { SpaceRadar } from "./SpaceRadar";
import { ShowcaseCardDetail, NeoAlertCardDetail } from "./SpaceOverlays";
import { buildShowcaseOrbitTrack } from "./neoShowcaseCatalog";
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

function AlertCardContent({
  ev,
  accent,
  cardTime,
  liveTick,
  neoBrief,
  neoLive,
  expanded,
  neoEnriched,
  trackLoading,
}: {
  ev: GlobeAlertEvent;
  accent: string;
  cardTime: string;
  liveTick: number;
  neoBrief?: EnrichedNeoBriefing;
  neoLive?: ReturnType<typeof liveNeoMetrics> | null;
  expanded?: boolean;
  neoEnriched?: EnrichedNeoBriefing | null;
  trackLoading?: boolean;
}) {
  void liveTick;
  const display = formatAlertCardDisplay(ev);

  return (
    <>
      <div className="global-alerts-card__headline" style={{ color: accent }}>
        {display.headline}
      </div>
      {display.chips.length > 0 ? (
        <div className="global-alerts-card__chips">
          {display.chips.map((chip) => (
            <span
              key={chip}
              className="global-alerts-card__chip"
              style={{ color: accent, borderColor: `${accent}55` }}
            >
              {chip}
            </span>
          ))}
        </div>
      ) : null}
      {display.region ? (
        <div className="global-alerts-card__region">
          <span className="global-alerts-card__label">אזור</span>
          <span
            className={`global-alerts-card__region-val${display.regionLtr ? " global-alerts-card__region-val--ltr" : ""}`}
          >
            {display.region}
          </span>
        </div>
      ) : null}
      {display.detail ? (
        <div className="global-alerts-card__detail-line">{display.detail}</div>
      ) : null}
      {ev.type === "neo" && neoBrief ? (
        <div className="global-alerts-card__detail-line">
          {neoBrief.impactRegion.split(" · ").pop()} · סכנה {neoBrief.riskLabel}
        </div>
      ) : null}
      {ev.type === "neo" && !ev.showcaseNeo ? (
        <div className="global-alerts-card__neo-countdown" aria-live="polite">
          <span className="global-alerts-card__neo-countdown-label">ספירה לקרבה / מעבר</span>
          <time className="global-alerts-card__neo-countdown-val">
            {formatNeoCountdown(ev.approachTime ?? ev.time)}
          </time>
        </div>
      ) : null}
      {expanded && ev.showcaseNeo ? <ShowcaseCardDetail ev={ev} /> : null}
      {expanded && ev.type === "neo" && !ev.showcaseNeo ? (
        <NeoAlertCardDetail
          ev={ev}
          neoEnriched={neoEnriched ?? neoBrief}
          neoLive={neoLive}
          loading={trackLoading}
        />
      ) : null}
      <div className="global-alerts-card__foot">
        <SeverityMeter ev={ev} color={accent} compact />
        <span className="global-alerts-card__time">
          {ev.type === "neo" && neoLive ? `${neoLive.speedKmS.toFixed(1)} km/s` : cardTime}
        </span>
      </div>
    </>
  );
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

function CollapsedEventPeek({
  ev,
  index,
  total,
  liveTick,
  onExpand,
}: {
  ev: GlobeAlertEvent;
  index: number;
  total: number;
  liveTick: number;
  onExpand: () => void;
}) {
  const accent = eventAccentColor(ev);
  const sev = getEventSeverity(ev);
  void liveTick;
  const display = formatAlertCardDisplay(ev);
  return (
    <button
      type="button"
      className="global-alerts-peek"
      onClick={onExpand}
      aria-label={`התרעה ${index + 1} מתוך ${total} — לחץ לפתיחת הרשימה`}
    >
      <span className="global-alerts-peek__accent" style={{ background: accent }} aria-hidden />
      <span className="global-alerts-peek__headline" style={{ color: accent }}>
        {display.headline}
      </span>
      {display.chips.length > 0 ? (
        <span className="global-alerts-peek__chips">{display.chips.slice(0, 3).join(" · ")}</span>
      ) : (
        <span className="global-alerts-peek__type" style={{ color: accent }}>
          {EVENT_TYPE_LABELS[ev.type].label}
        </span>
      )}
      {display.region ? (
        <span
          className={`global-alerts-peek__loc${display.regionLtr ? " global-alerts-peek__loc--ltr" : ""}`}
        >
          {display.region}
        </span>
      ) : null}
      <SeverityMeter ev={ev} color={accent} compact />
      <span className="global-alerts-peek__time">
        {ev.type === "neo" ? formatNeoCountdown(ev.approachTime ?? ev.time) : timeAgo(ev.time)}
      </span>
      {total > 1 ? (
        <span className="global-alerts-peek__dots" aria-hidden>
          {Array.from({ length: total }, (_, i) => (
            <span key={i} className={`global-alerts-peek__dot${i === index ? " is-active" : ""}`} />
          ))}
        </span>
      ) : null}
      <span className={`global-alerts-peek__tier is-sev-${sev.tier}`} aria-hidden />
    </button>
  );
}

type SheetMode = "collapsed" | "list" | "detail";

function TimeRangeSelect({
  value,
  onChange,
}: {
  value: AlertTimeRange;
  onChange: (v: AlertTimeRange) => void;
}) {
  return (
    <div className="global-alerts-range global-alerts-range--dual">
      {ALERT_TIME_RANGE_OPTIONS.map((opt) => (
        <button
          key={opt.id}
          type="button"
          className={`global-alerts-range__btn${value === opt.id ? " is-active" : ""}`}
          onClick={() => onChange(opt.id)}
          aria-pressed={value === opt.id}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function GlobalAlertsPanel({ onClose }: Props) {
  const sceneRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<GlobeSceneHandle | null>(null);
  const [sheetMode, setSheetMode] = useState<SheetMode>("collapsed");
  const [rotateIdx, setRotateIdx] = useState(0);
  const [sceneReady, setSceneReady] = useState(false);
  const [focused, setFocused] = useState<GlobeAlertEvent | null>(null);
  const focusedRef = useRef<GlobeAlertEvent | null>(null);
  focusedRef.current = focused;
  const [stormEnriched, setStormEnriched] = useState<EnrichedStormBriefing | null>(null);
  const [trackLoading, setTrackLoading] = useState(false);
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 768px)").matches,
  );
  const [neoTracks, setNeoTracks] = useState<Record<string, import("./neoTrack").NeoOrbitTrack>>({});
  const [neoEnriched, setNeoEnriched] = useState<EnrichedNeoBriefing | null>(null);
  const [neoBriefCache, setNeoBriefCache] = useState<Record<string, EnrichedNeoBriefing>>({});
  const [liveTick, setLiveTick] = useState(0);
  const [timeRange, setTimeRange] = useState<AlertTimeRange>("live");
  const { events, loading, lastUpdate } = useGlobalAlertEvents(true, timeRange);

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

  const panelCollapsed = sheetMode === "collapsed";
  const showHeaderFocus = isMobile && !!focused && panelCollapsed;
  const showSheetDetail = !isMobile && sheetMode === "detail" && !!focused;

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

  const loadNeoTrack = useCallback(async (ev: GlobeAlertEvent, spaceMode = false) => {
    if (ev.type !== "neo") {
      globeRef.current?.clearNeoTrack();
      return;
    }
    globeRef.current?.clearStormTrack();
    setStormEnriched(null);

    if (ev.showcaseNeo) {
      const track = buildShowcaseOrbitTrack(ev);
      setNeoTracks((prev) => ({ ...prev, [ev.id]: track }));
      setTrackLoading(false);
      setNeoEnriched(null);
      return;
    }

    setTrackLoading(true);
    const briefPromise = enrichNeoBriefing(ev);

    let track = ev.designation
      ? await fetchNeoHorizonsTrack(ev.designation, ev.approachTime ?? ev.time, 14).catch(() => null)
      : null;
    if (!track || track.points.length < 2) {
      track = buildSyntheticNeoTrack(ev);
    }

    const brief = await briefPromise;
    setNeoEnriched(brief);
    setNeoBriefCache((prev) => ({ ...prev, [ev.id]: brief }));
    setTrackLoading(false);
    setNeoTracks((prev) => ({ ...prev, [ev.id]: track! }));

    if (spaceMode) {
      globeRef.current?.clearNeoTrack();
      globeRef.current?.focusSpaceNeo(ev);
      return;
    }

    globeRef.current?.focusEvent(ev);
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
        await loadNeoTrack(ev, timeRange === "space");
        return;
      }
      setStormEnriched(null);
      setTrackLoading(false);
      globeRef.current?.clearStormTrack();
      globeRef.current?.clearNeoTrack();
    },
    [loadStormBriefing, loadNeoTrack, timeRange],
  );

  const handleCardClick = useCallback(
    (ev: GlobeAlertEvent) => {
      if (focusedRef.current?.id === ev.id && timeRange === "space" && ev.type === "neo") {
        setFocused(null);
        globeRef.current?.returnToNormal();
        return;
      }
      setFocused(ev);
      if (timeRange === "space" && ev.type === "neo") {
        setSheetMode("list");
        globeRef.current?.focusSpaceNeo(ev);
      } else if (ev.showcaseNeo) {
        setSheetMode("list");
      } else {
        setSheetMode(isMobile ? "collapsed" : "detail");
      }
      if (timeRange !== "space" && !ev.showcaseNeo) {
        globeRef.current?.focusEvent(ev);
      }
      void loadEventFocus(ev);
    },
    [loadEventFocus, isMobile, timeRange],
  );

  const handleCardClickRef = useRef(handleCardClick);
  handleCardClickRef.current = handleCardClick;

  const handleReturn = useCallback(() => {
    setFocused(null);
    setStormEnriched(null);
    setNeoEnriched(null);
    setTrackLoading(false);
    globeRef.current?.returnToNormal();
    if (!isMobile) {
      setSheetMode((mode) => (mode === "detail" ? "list" : mode));
    }
  }, [isMobile]);

  useEffect(() => {
    if (!sceneRef.current) return;
    globeRef.current = initGlobeScene(sceneRef.current, {
      onLoaded: () => setSceneReady(true),
      onFocus: (ev) => setFocused(ev),
      onEventPick: (ev) => handleCardClickRef.current(ev),
    });
    return () => {
      globeRef.current?.dispose();
      globeRef.current = null;
    };
  }, []);

  const alertEvents = useMemo(() => {
    if (timeRange === "space") {
      return filterSpaceAlerts(events);
    }
    return events.filter((e) => e.type !== "neo" && e.type !== "fireball");
  }, [events, timeRange]);

  const globeEvents = useMemo(() => {
    if (timeRange === "space") return events;
    return alertEvents;
  }, [events, timeRange, alertEvents]);

  const showcaseEvents = useMemo(() => {
    if (timeRange !== "space") return [];
    return events
      .filter((e) => e.showcaseNeo)
      .sort((a, b) => (a.distLd ?? 99) - (b.distLd ?? 99));
  }, [events, timeRange]);

  useEffect(() => {
    if (!globeRef.current || !sceneReady) return;
    globeRef.current.setSpaceMode(timeRange === "space");
    globeRef.current.syncEvents(globeEvents);
  }, [globeEvents, sceneReady, timeRange]);

  useEffect(() => {
    globeRef.current?.setNeoOrbitTracks(neoTracks);
  }, [neoTracks, sceneReady, timeRange]);

  useEffect(() => {
    if (timeRange !== "space") return;
    const showcases = events.filter((e) => e.showcaseNeo);
    if (!showcases.length) return;
    setNeoTracks((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const ev of showcases) {
        if (!next[ev.id]) {
          next[ev.id] = buildShowcaseOrbitTrack(ev);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [timeRange, events]);

  useEffect(() => {
    if (timeRange !== "space") return;
    const neos = events.filter((e) => e.type === "neo");
    if (!neos.length) return;
    let cancelled = false;
    (async () => {
      const batch: Record<string, import("./neoTrack").NeoOrbitTrack> = {};
      for (const neo of neos) {
        if (cancelled) return;
        if (neoTracks[neo.id]) continue;
        if (neo.showcaseNeo) {
          batch[neo.id] = buildShowcaseOrbitTrack(neo);
          continue;
        }
        if (!neo.designation) {
          batch[neo.id] = buildSyntheticNeoTrack(neo);
          continue;
        }
        let track = await fetchNeoHorizonsTrack(neo.designation, neo.approachTime ?? neo.time, 14).catch(
          () => null,
        );
        if (!track || track.points.length < 2) {
          track = buildSyntheticNeoTrack(neo);
        }
        batch[neo.id] = track;
      }
      if (cancelled || !Object.keys(batch).length) return;
      setNeoTracks((prev) => {
        const next = { ...prev };
        for (const [id, tr] of Object.entries(batch)) {
          if (!next[id]) next[id] = tr;
        }
        return next;
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [timeRange, events]);

  const listEvents = alertEvents;

  useEffect(() => {
    setFocused(null);
    setStormEnriched(null);
    setNeoEnriched(null);
    setNeoTracks({});
    globeRef.current?.returnToNormal();
    setSheetMode("list");
  }, [timeRange]);

  const rotationEvents = useMemo(() => {
    return [...listEvents].sort((a, b) => {
      const sa = getEventSeverity(a).score;
      const sb = getEventSeverity(b).score;
      if (sa !== sb) return sb - sa;
      return b.time - a.time;
    });
  }, [listEvents]);

  useEffect(() => {
    setRotateIdx(0);
  }, [rotationEvents]);

  useEffect(() => {
    if (!panelCollapsed || rotationEvents.length < 2) return;
    const id = window.setInterval(() => {
      setRotateIdx((i) => (i + 1) % rotationEvents.length);
    }, 4500);
    return () => window.clearInterval(id);
  }, [panelCollapsed, rotationEvents.length]);

  const peekEvent = rotationEvents[rotateIdx] ?? rotationEvents[0] ?? null;

  useEffect(() => {
    if (!focused || focused.type !== "neo") return;
    const el = document.getElementById(`space-card-${focused.id}`);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [focused]);

  useEffect(() => {
    if (timeRange !== "space") return;
    const neos = events.filter((e) => e.type === "neo" && !e.showcaseNeo);
    if (!neos.length) return;
    let cancelled = false;
    (async () => {
      for (const neo of neos.slice(0, 12)) {
        if (cancelled || neoBriefCache[neo.id]) continue;
        const brief = await enrichNeoBriefing(neo);
        if (cancelled) return;
        setNeoBriefCache((prev) => (prev[neo.id] ? prev : { ...prev, [neo.id]: brief }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [timeRange, events]);

  useEffect(() => {
    if (!focused || focused.type !== "hurricane") return;
    const id = window.setInterval(() => {
      void loadEventFocus(focused);
    }, 60_000);
    return () => window.clearInterval(id);
  }, [focused, loadEventFocus]);

  const renderEventCard = (ev: GlobeAlertEvent) => {
    const accent = eventAccentColor(ev);
    const isActive = focused?.id === ev.id;
    const sev = getEventSeverity(ev);
    const track = ev.type === "neo" ? neoTracks[ev.id] : undefined;
    const live = ev.type === "neo" ? liveNeoMetrics(ev, track) : null;
    const brief = ev.type === "neo" ? neoBriefCache[ev.id] : undefined;
    const cardTime =
      ev.source === "gdacs" && ev.gdacsIsCurrent !== false
        ? "● פעיל"
        : ev.source === "gdacs"
          ? timeAgo(ev.updatedTime ?? ev.gdacsStartTime ?? ev.time)
          : timeAgo(ev.time);
    return (
      <div
        key={`${timeRange}-${ev.id}`}
        id={`space-card-${ev.id}`}
        className={`global-alerts-card is-sev-${sev.tier}${isActive ? " is-active" : ""}${isActive && ev.type === "neo" ? " is-showcase-expanded" : ""}`}
        style={{ borderRightColor: accent, "--sev-color": accent } as CSSProperties}
        onClick={() => handleCardClick(ev)}
        onKeyDown={(e) => e.key === "Enter" && handleCardClick(ev)}
        role="button"
        tabIndex={0}
      >
        <div className="global-alerts-card__inner">
          <AlertCardContent
            ev={ev}
            accent={accent}
            cardTime={cardTime}
            liveTick={liveTick}
            neoBrief={brief}
            neoLive={live}
            expanded={isActive && ev.type === "neo"}
            neoEnriched={isActive ? neoEnriched : undefined}
            trackLoading={isActive && trackLoading}
          />
        </div>
      </div>
    );
  };

  const updateLabel = lastUpdate
    ? new Date(lastUpdate).toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" })
    : "--:--";

  const showLoading = loading && !sceneReady;
  const sheetExpanded = sheetMode !== "collapsed";

  const toggleSheet = useCallback(() => {
    setSheetMode((mode) => {
      if (mode === "collapsed") return "list";
      if (mode === "detail") return "list";
      return "collapsed";
    });
  }, []);

  const openList = useCallback(() => {
    setSheetMode("list");
  }, []);

  const isSpaceMode = timeRange === "space";

  return (
    <div
      className={`global-alerts-root${isMobile ? "" : " global-alerts-root--desktop"}${sheetExpanded ? " global-alerts-root--sheet-expanded" : ""}${showSheetDetail ? " global-alerts-root--sheet-detail" : ""}`}
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
            ) : focused.type === "neo" && isSpaceMode ? (
              <EventFocusStrip ev={focused} onReturn={handleReturn} />
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
          ) : !isSpaceMode ? (
            <p className="global-alerts-header__hint">
              <span className="global-alerts-hdot" style={{ background: "#3498db" }} /> סיבוב
              <span className="global-alerts-hdot" style={{ background: "#3498db" }} /> זום
              <span className="global-alerts-hdot" style={{ background: "#ffeb3b" }} /> לחץ על אירוע
            </p>
          ) : null}
        </div>
      </header>

      <aside
        className={`global-alerts-sheet${panelCollapsed ? " is-collapsed" : sheetMode === "detail" ? " is-detail" : " is-expanded"}`}
        aria-label="התרעות"
      >
        <div className="global-alerts-sheet__handle">
          <button
            type="button"
            className="global-alerts-sheet__grab"
            onClick={toggleSheet}
            aria-expanded={sheetExpanded}
            aria-label={
              sheetMode === "detail"
                ? "חזרה לרשימת התרעות"
                : sheetExpanded
                  ? "כווץ לשורת התרעה"
                  : "פתח רשימת התרעות"
            }
          >
            <span className="global-alerts-sheet__pill" aria-hidden />
          </button>
          {panelCollapsed ? (
            peekEvent ? (
              <CollapsedEventPeek
                ev={peekEvent}
                index={rotateIdx}
                total={rotationEvents.length}
                liveTick={liveTick}
                onExpand={openList}
              />
            ) : (
              <button
                type="button"
                className="global-alerts-peek global-alerts-peek--empty"
                onClick={openList}
              >
                <span className="global-alerts-peek__type">התרעות פעילות</span>
                <span className="global-alerts-peek__loc">אין אירועים פעילים כרגע</span>
              </button>
            )
          ) : sheetMode === "detail" && focused ? (
            <div className="global-alerts-sheet__title-row global-alerts-sheet__title-row--detail">
              <button type="button" className="global-alerts-sheet__back" onClick={handleReturn}>
                ← רשימה
              </button>
            </div>
          ) : (
            <div className="global-alerts-sheet__title-row">
              <h2>{rangeLabel(timeRange)}</h2>
              <span className="global-alerts-sheet__count">{listEvents.length}</span>
            </div>
          )}
          <button
            type="button"
            className="global-alerts-sheet__chevron-btn"
            onClick={toggleSheet}
            aria-label={sheetMode === "detail" ? "חזרה לרשימה" : sheetExpanded ? "כווץ" : "הרחב"}
          >
            {sheetMode === "detail" ? "▲" : sheetExpanded ? "▼" : "▲"}
          </button>
        </div>

        {showSheetDetail && focused ? (
          <div className="global-alerts-sheet__detail">
            {focused.type === "hurricane" ? (
              <HurricaneBriefingCard
                ev={focused}
                enriched={stormEnriched}
                loading={trackLoading}
                onReturn={handleReturn}
              />
            ) : (
              <EventFocusDock
                ev={focused}
                enriched={stormEnriched}
                loading={trackLoading}
                neoTrack={focused.type === "neo" ? neoTracks[focused.id] ?? null : null}
                neoEnriched={focused.type === "neo" ? neoEnriched : null}
                onReturn={handleReturn}
              />
            )}
          </div>
        ) : !panelCollapsed ? (
          <>
            <div className="global-alerts-panel__header">
              <TimeRangeSelect value={timeRange} onChange={setTimeRange} />
              {timeRange === "live" ? (
                <p className="global-alerts-panel__range-note">
                  כל ההתרעות הפעילות · רעידות מ-30 דק&apos; אחרונות
                </p>
              ) : (
                <p className="global-alerts-panel__range-note">
                  התרעות בזמן אמת · NASA CAD 14 יום (קרבה קרובה)
                </p>
              )}
            </div>

        <div className="global-alerts-list">
          {loading ? (
            <p className="global-alerts-empty global-alerts-empty--loading">טוען התרעות…</p>
          ) : null}
          {!listEvents.length && !loading ? (
            <p className="global-alerts-empty">
              {timeRange === "live"
                ? "אין התרעות פעילות כרגע."
                : "אין אסטרואידים ב-14 הימים הקרובים."}
            </p>
          ) : null}
          {listEvents.slice(0, 50).map((ev) => renderEventCard(ev))}
          {showcaseEvents.length ? (
            <>
              <div className="global-alerts-list__section">
                <span>אובייקטים מחזוריים</span>
                <span className="global-alerts-list__section-count">{showcaseEvents.length}</span>
              </div>
              {showcaseEvents.map((ev) => renderEventCard(ev))}
            </>
          ) : null}
        </div>

        <div className="global-alerts-panel__footer">
          <span>
            {timeRange === "live" ? (
              <>
                <span className="global-alerts-sdot" />
                חי
              </>
            ) : (
              "חלל · עתידי"
            )}
          </span>
          <span>{updateLabel}</span>
        </div>
          </>
        ) : null}
      </aside>

      {isSpaceMode && focused && !focused.showcaseNeo ? (
        <button type="button" className="space-return-btn" onClick={handleReturn}>
          חזרה
        </button>
      ) : null}

      <SpaceRadar events={globeEvents} focusedId={focused?.id} visible={isSpaceMode} />

      <div ref={sceneRef} className="global-alerts-scene" />
    </div>
  );
}
