import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "../searchResults/types";
import { fetchStartupContext, getStartupContextSync, refreshLocalWeather } from "../startupContext";
import type { StartupContext } from "../startupContext/types";
import { CableStreamSlot } from "./CableStreamSlot";
import { formatCableOsdDate, formatCableOsdWeather, shortenCableChannelTitle } from "./cableOsdContext";
import {
  CABLE_STREAM_LOAD_MS,
  CABLE_WARM_SWITCH_MS,
  QUAD_ROTATE_MS,
  advanceQuadRotation,
  cableOsdRangeLabel,
  favoriteForPage,
  initialQuadSlots,
  initialRotationCursor,
  isQuadPage,
  maxCablePageIndex,
  nextCablePageIndex,
  nextFavoriteIndex,
  nextWorkingFavoriteIndex,
  pageIndexForFavorite,
  pickCableQuadFromSlots,
  prevFavoriteIndex,
  singleFavoriteIndex,
  targetFavoriteAfterStep,
} from "./cableTunerUtils";
import "./cableTuner.css";

const TUNE_MS = 2400;
const OSD_HIDE_MS = 4500;
const VOLUME_KEY = "grovee-cable-volume";

function readStoredVolume(): number {
  try {
    const raw = localStorage.getItem(VOLUME_KEY);
    if (raw == null) return 0.75;
    const n = Number(raw);
    return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0.75;
  } catch {
    return 0.75;
  }
}

type Props = {
  favorites: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  loading: boolean;
  onOpenBrowse: () => void;
  onRemoveFavorite: (hit: UnifiedSearchHit) => void | Promise<void>;
};

function formatClock(date: Date, rtl: boolean, timezone?: string): string {
  const locale = rtl ? "he-IL" : "en-US";
  try {
    return date.toLocaleTimeString(locale, {
      timeZone: timezone,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return date.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }
}

export function CableTunerView({ favorites, uiLang, loading, onOpenBrowse, onRemoveFavorite }: Props) {
  const rtl = uiLang === "he";
  const [pageIndex, setPageIndex] = useState(0);
  const [quadSlots, setQuadSlots] = useState<number[]>(() => initialQuadSlots(favorites.length));
  const [rotationCursor, setRotationCursor] = useState(() => initialRotationCursor(favorites.length));
  const [selectedQuadSlot, setSelectedQuadSlot] = useState(0);
  const rotationSlotRef = useRef(0);
  const quadStateRef = useRef({ slots: initialQuadSlots(favorites.length), cursor: initialRotationCursor(favorites.length) });
  const [globalSnow, setGlobalSnow] = useState(false);
  const [osdVisible, setOsdVisible] = useState(true);
  const [confirmRemoveOpen, setConfirmRemoveOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [localCtx, setLocalCtx] = useState<StartupContext | null>(() => getStartupContextSync());
  const [now, setNow] = useState(() => new Date());
  const [volume, setVolume] = useState(readStoredVolume);
  const [userMuted, setUserMuted] = useState(false);
  const [audioUnlocked, setAudioUnlocked] = useState(false);
  const [preloadFavIdx, setPreloadFavIdx] = useState(0);
  const [preloadReady, setPreloadReady] = useState(false);
  const [deadFavorites, setDeadFavorites] = useState<ReadonlySet<number>>(() => new Set());
  const tuneTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const osdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  const L =
    uiLang === "he"
      ? {
          noFavs: "אין ערוצים במועדפים — פתח חיפוש והוסף ☆ לערוצים שעובדים.",
          browse: "חיפוש",
          chUp: "▲",
          chDown: "▼",
          tuning: "מכוון…",
          range: (a: number, b: number) => (a === b ? `ערוץ ${a}` : `ערוצים ${a}–${b}`),
          fullscreen: "מסך מלא",
          exitFullscreen: "צא ממסך מלא",
          removeFav: "הסר מהמועדפים",
          confirmTitle: "להסיר מהמועדפים?",
          confirmBody: "הערוץ יוסר מרשימת המועדפים שלך. אפשר תמיד להוסיף אותו שוב מחיפוש הערוצים.",
          confirmYes: "כן, הסר",
          confirmNo: "ביטול",
          volume: "עוצמת קול",
          mute: "השתק",
          unmute: "הפעל שמע",
        }
      : {
          noFavs: "No favorites — open search and star ☆ working channels.",
          browse: "Search",
          chUp: "▲",
          chDown: "▼",
          tuning: "Tuning…",
          range: (a: number, b: number) => (a === b ? `CH ${a}` : `CH ${a}–${b}`),
          fullscreen: "Fullscreen",
          exitFullscreen: "Exit fullscreen",
          removeFav: "Remove favorite",
          confirmTitle: "Remove from favorites?",
          confirmBody: "This channel will be removed from your favorites. You can add it again from channel search.",
          confirmYes: "Yes, remove",
          confirmNo: "Cancel",
          volume: "Volume",
          mute: "Mute",
          unmute: "Unmute",
        };

  const total = favorites.length;
  const showQuad = isQuadPage(pageIndex);
  const quadTiles = useMemo(() => pickCableQuadFromSlots(favorites, quadSlots), [favorites, quadSlots]);
  const singleHit = useMemo(() => favoriteForPage(favorites, pageIndex), [favorites, pageIndex]);
  const focusHit = showQuad ? quadTiles[selectedQuadSlot] ?? null : singleHit;

  const preloadHit = useMemo(() => {
    if (showQuad || total < 2) return null;
    return favorites[preloadFavIdx % total] ?? null;
  }, [favorites, preloadFavIdx, showQuad, total]);

  const warmSwitchTarget = useMemo(() => {
    if (showQuad || total < 2) return -1;
    return targetFavoriteAfterStep(pageIndex, 1, total, deadFavorites);
  }, [deadFavorites, pageIndex, showQuad, total]);

  const markFavoriteDead = useCallback((favoriteIndex: number) => {
    setDeadFavorites((prev) => {
      if (prev.has(favoriteIndex)) return prev;
      const next = new Set(prev);
      next.add(favoriteIndex);
      return next;
    });
  }, []);

  const range = cableOsdRangeLabel(pageIndex, total, showQuad ? quadSlots : undefined);
  const rangeLabel = range ? L.range(range.from, range.to) : "";
  const dateLabel = formatCableOsdDate(now, uiLang, localCtx?.timezone);
  const weatherLabel = formatCableOsdWeather(localCtx);
  const clock = formatClock(now, rtl, localCtx?.timezone);
  const centerTitle = focusHit ? shortenCableChannelTitle(focusHit.title) : "";

  const pokeOsd = useCallback(() => {
    setOsdVisible(true);
    if (osdTimer.current) clearTimeout(osdTimer.current);
    if (!confirmRemoveOpen) {
      osdTimer.current = setTimeout(() => setOsdVisible(false), OSD_HIDE_MS);
    }
  }, [confirmRemoveOpen]);

  const closeConfirmRemove = useCallback(() => {
    setConfirmRemoveOpen(false);
    pokeOsd();
  }, [pokeOsd]);

  const openConfirmRemove = useCallback(() => {
    if (!focusHit) return;
    if (osdTimer.current) clearTimeout(osdTimer.current);
    setOsdVisible(true);
    setConfirmRemoveOpen(true);
  }, [focusHit]);

  const clearTuneTimer = useCallback(() => {
    if (tuneTimer.current) clearTimeout(tuneTimer.current);
    tuneTimer.current = null;
  }, []);

  const handleQuadSlotFail = useCallback(
    (slotIndex: number) => {
      if (total < 2) return;
      const advanced = advanceQuadRotation(quadStateRef.current.slots, slotIndex, quadStateRef.current.cursor, total);
      quadStateRef.current = advanced;
      setQuadSlots(advanced.slots);
      setRotationCursor(advanced.cursor);
    },
    [total],
  );

  const advancePreloadCandidate = useCallback(() => {
    setPreloadReady(false);
    setPreloadFavIdx((idx) => {
      markFavoriteDead(idx);
      return nextWorkingFavoriteIndex(idx, 1, total, new Set([...deadFavorites, idx]));
    });
  }, [deadFavorites, markFavoriteDead, total]);

  const skipSingleToNext = useCallback(() => {
    if (showQuad || total < 2 || globalSnow) return;
    const cur = singleFavoriteIndex(pageIndex);
    markFavoriteDead(cur);
    const nextFav = nextWorkingFavoriteIndex(cur, 1, total, new Set([...deadFavorites, cur]));
    const targetPage = pageIndexForFavorite(nextFav);
    clearTuneTimer();
    setGlobalSnow(true);
    tuneTimer.current = setTimeout(() => {
      setPageIndex(targetPage);
      setGlobalSnow(false);
    }, TUNE_MS);
  }, [clearTuneTimer, deadFavorites, globalSnow, markFavoriteDead, pageIndex, showQuad, total]);

  const changePage = useCallback(
    (delta: 1 | -1) => {
      if (total < 1 || globalSnow) return;
      if (total <= 1) return;
      clearTuneTimer();
      pokeOsd();

      let targetPage: number;
      if (isQuadPage(pageIndex)) {
        const targetFav = targetFavoriteAfterStep(pageIndex, delta, total, deadFavorites);
        targetPage = pageIndexForFavorite(targetFav);
      } else {
        const curFav = singleFavoriteIndex(pageIndex);
        if (delta === -1 && curFav === 0) {
          targetPage = 0;
        } else if (delta === 1 && pageIndex === total) {
          targetPage = 0;
        } else {
          const targetFav = nextWorkingFavoriteIndex(curFav, delta, total, deadFavorites);
          targetPage = pageIndexForFavorite(targetFav);
        }
      }

      const targetFav = isQuadPage(targetPage) ? -1 : singleFavoriteIndex(targetPage);
      const warmHit = !isQuadPage(targetPage) && preloadReady && targetFav >= 0 && preloadFavIdx === targetFav;
      const tuneMs = warmHit ? CABLE_WARM_SWITCH_MS : TUNE_MS;
      setGlobalSnow(true);
      tuneTimer.current = setTimeout(() => {
        setPageIndex(targetPage);
        setGlobalSnow(false);
        if (warmHit) setPreloadReady(false);
      }, tuneMs);
    },
    [clearTuneTimer, deadFavorites, globalSnow, pageIndex, pokeOsd, preloadFavIdx, preloadReady, total],
  );

  const toggleFullscreen = useCallback(async () => {
    const el = rootRef.current;
    if (!el) return;
    pokeOsd();
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await el.requestFullscreen();
      }
    } catch {
      /* unsupported */
    }
  }, [pokeOsd]);

  const handleRemoveFavorite = useCallback(async () => {
    if (!focusHit) return;
    await onRemoveFavorite(focusHit);
    setConfirmRemoveOpen(false);
    if (showQuad) setSelectedQuadSlot(0);
    pokeOsd();
  }, [focusHit, onRemoveFavorite, pokeOsd, showQuad]);

  useEffect(() => {
    let alive = true;
    void fetchStartupContext().then(async (ctx) => {
      const withWx = await refreshLocalWeather(ctx);
      if (alive) setLocalCtx(withWx);
    });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const slots = initialQuadSlots(total);
    const cursor = initialRotationCursor(total);
    quadStateRef.current = { slots, cursor };
    setQuadSlots(slots);
    setRotationCursor(cursor);
    rotationSlotRef.current = 0;
    setSelectedQuadSlot(0);
  }, [total]);

  useEffect(() => {
    setPageIndex((p) => Math.min(p, maxCablePageIndex(total)));
  }, [total]);

  useEffect(() => {
    if (showQuad || total < 2) return;
    setPreloadFavIdx(targetFavoriteAfterStep(pageIndex, 1, total, deadFavorites));
    setPreloadReady(false);
  }, [deadFavorites, pageIndex, showQuad, total]);

  useEffect(() => {
    setDeadFavorites(new Set());
  }, [favorites]);

  useEffect(() => {
    if (!showQuad || total < 2) return;
    const id = window.setInterval(() => {
      const slot = rotationSlotRef.current % 4;
      rotationSlotRef.current += 1;
      const advanced = advanceQuadRotation(quadStateRef.current.slots, slot, quadStateRef.current.cursor, total);
      quadStateRef.current = advanced;
      setQuadSlots(advanced.slots);
      setRotationCursor(advanced.cursor);
      pokeOsd();
    }, QUAD_ROTATE_MS);
    return () => window.clearInterval(id);
  }, [showQuad, total, pokeOsd]);

  useEffect(() => {
    pokeOsd();
    const tick = setInterval(() => setNow(new Date()), 1000);
    return () => {
      clearInterval(tick);
      if (osdTimer.current) clearTimeout(osdTimer.current);
    };
  }, [pokeOsd]);

  useEffect(() => {
    const onFs = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (confirmRemoveOpen) {
        if (e.key === "Escape") {
          e.preventDefault();
          closeConfirmRemove();
        }
        return;
      }
      pokeOsd();
      if (e.key === "ArrowUp" || e.key === "PageUp") {
        e.preventDefault();
        changePage(1);
      }
      if (e.key === "ArrowDown" || e.key === "PageDown") {
        e.preventDefault();
        changePage(-1);
      }
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        void toggleFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [changePage, closeConfirmRemove, confirmRemoveOpen, pokeOsd, toggleFullscreen]);

  useEffect(() => () => clearTuneTimer(), [clearTuneTimer]);

  const onPointerActivity = useCallback(() => {
    pokeOsd();
    if (!audioUnlocked) setAudioUnlocked(true);
  }, [audioUnlocked, pokeOsd]);

  const slotMuted = useCallback(
    (audioFocus: boolean) => !audioUnlocked || userMuted || !audioFocus,
    [audioUnlocked, userMuted],
  );

  const onVolumeChange = useCallback((next: number) => {
    const clamped = Math.max(0, Math.min(1, next));
    setVolume(clamped);
    try {
      localStorage.setItem(VOLUME_KEY, String(clamped));
    } catch {
      /* ignore */
    }
    if (clamped > 0) setUserMuted(false);
  }, []);

  if (loading) {
    return <div className="lm-cable-loading">…</div>;
  }

  if (!total) {
    return (
      <div className="lm-cable-empty">
        <p>{L.noFavs}</p>
        <button type="button" className="lm-cable-osd-btn" onClick={onOpenBrowse}>
          {L.browse}
        </button>
      </div>
    );
  }

  return (
    <div
      ref={rootRef}
      className={`lm-cable lm-cable--fullscreen${isFullscreen ? " is-browser-fullscreen" : ""}`}
      dir={rtl ? "rtl" : "ltr"}
      onMouseMove={onPointerActivity}
      onTouchStart={onPointerActivity}
    >
      {showQuad ? (
        <div className="lm-cable-grid lm-cable-grid--4">
          {Array.from({ length: 4 }, (_, i) => {
            const hit = quadTiles[i] ?? null;
            const audioFocus = selectedQuadSlot === i;
            return (
              <CableStreamSlot
                key={`quad-slot-${i}`}
                hit={hit}
                globalSnow={globalSnow}
                osdVisible={osdVisible}
                channelNum={quadSlots[i] != null ? quadSlots[i] + 1 : 0}
                selected={audioFocus}
                audioFocus={audioFocus && audioUnlocked && !userMuted}
                muted={slotMuted(audioFocus)}
                volume={volume}
                multiView
                loadTimeoutMs={CABLE_STREAM_LOAD_MS}
                onStreamFail={() => handleQuadSlotFail(i)}
                onSelect={() => {
                  setSelectedQuadSlot(i);
                  setAudioUnlocked(true);
                  pokeOsd();
                }}
              />
            );
          })}
        </div>
      ) : singleHit ? (
        <div className="lm-cable-single">
          <CableStreamSlot
            key={`${singleHit.id}-s-${pageIndex}`}
            hit={singleHit}
            globalSnow={globalSnow}
            osdVisible={osdVisible}
            channelNum={singleFavoriteIndex(pageIndex) + 1}
            audioFocus={audioUnlocked && !userMuted}
            muted={slotMuted(true)}
            volume={volume}
            single
            loadTimeoutMs={CABLE_STREAM_LOAD_MS}
            onStreamFail={skipSingleToNext}
          />
        </div>
      ) : null}

      {!showQuad ? (
        <div className={`lm-cable-top-bar${osdVisible && !confirmRemoveOpen ? "" : " is-hidden"}`}>
          <button
            type="button"
            className="lm-cable-top-btn lm-cable-top-btn--danger"
            onClick={openConfirmRemove}
            disabled={!focusHit}
            aria-label={L.removeFav}
            title={L.removeFav}
          >
            <span className="lm-cable-top-btn-icon" aria-hidden="true">
              ✕
            </span>
            <span className="lm-cable-top-btn-label">{L.removeFav}</span>
          </button>
        </div>
      ) : null}

      {confirmRemoveOpen && focusHit && !showQuad ? (
        <div className="lm-cable-confirm-backdrop" onClick={closeConfirmRemove} role="presentation">
          <div
            className="lm-cable-confirm"
            role="dialog"
            aria-modal="true"
            aria-labelledby="lm-cable-confirm-title"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="lm-cable-confirm-kicker">{L.removeFav}</p>
            <h3 id="lm-cable-confirm-title" className="lm-cable-confirm-title">
              {L.confirmTitle}
            </h3>
            <p className="lm-cable-confirm-channel">{focusHit.title}</p>
            <p className="lm-cable-confirm-body">{L.confirmBody}</p>
            <div className="lm-cable-confirm-actions">
              <button type="button" className="lm-cable-confirm-btn lm-cable-confirm-btn--ghost" onClick={closeConfirmRemove}>
                {L.confirmNo}
              </button>
              <button
                type="button"
                className="lm-cable-confirm-btn lm-cable-confirm-btn--danger"
                onClick={() => void handleRemoveFavorite()}
              >
                {L.confirmYes}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div className={`lm-cable-osd${osdVisible ? "" : " is-hidden"}`} dir="ltr">
        <div className="lm-cable-osd-actions">
          <button
            type="button"
            className="lm-cable-osd-btn lm-cable-osd-btn--remote"
            onClick={() => changePage(1)}
            disabled={globalSnow || total <= 1}
            aria-label={L.chUp}
          >
            {L.chUp}
          </button>
          <button
            type="button"
            className="lm-cable-osd-btn lm-cable-osd-btn--remote"
            onClick={() => changePage(-1)}
            disabled={globalSnow || total <= 1}
            aria-label={L.chDown}
          >
            {L.chDown}
          </button>
          <button type="button" className="lm-cable-osd-btn" onClick={onOpenBrowse}>
            {L.browse}
          </button>
          <button
            type="button"
            className="lm-cable-osd-btn lm-cable-osd-btn--icon"
            onClick={() => void toggleFullscreen()}
            aria-label={isFullscreen ? L.exitFullscreen : L.fullscreen}
            title={isFullscreen ? L.exitFullscreen : L.fullscreen}
          >
            {isFullscreen ? "⤢" : "⛶"}
          </button>
          <div className="lm-cable-volume" title={L.volume}>
            <button
              type="button"
              className="lm-cable-osd-btn lm-cable-osd-btn--icon lm-cable-volume-mute"
              onClick={() => {
                setAudioUnlocked(true);
                setUserMuted((m) => !m);
                pokeOsd();
              }}
              aria-label={userMuted ? L.unmute : L.mute}
            >
              {userMuted || volume === 0 ? "🔇" : volume < 0.45 ? "🔉" : "🔊"}
            </button>
            <input
              type="range"
              className="lm-cable-volume-slider"
              min={0}
              max={100}
              value={Math.round((userMuted ? 0 : volume) * 100)}
              onChange={(e) => {
                setAudioUnlocked(true);
                onVolumeChange(Number(e.target.value) / 100);
                pokeOsd();
              }}
              aria-label={L.volume}
            />
          </div>
        </div>

        <div className={`lm-cable-osd-center${showQuad ? " lm-cable-osd-center--quad" : ""}`} dir={rtl ? "rtl" : "ltr"}>
          {!showQuad ? (
            <h2 className="lm-cable-osd-channel" title={focusHit?.title}>
              {centerTitle}
              {rangeLabel ? <span className="lm-cable-osd-inline-meta"> · {rangeLabel}</span> : null}
            </h2>
          ) : null}
          {globalSnow ? <span className="lm-cable-osd-tuning">{L.tuning}</span> : null}
        </div>

        <div className="lm-cable-osd-right">
          <span className="lm-cable-osd-clock">{clock}</span>
          <span className="lm-cable-osd-date">{dateLabel}</span>
          {weatherLabel ? <span className="lm-cable-osd-weather">{weatherLabel}</span> : null}
        </div>
      </div>

      {!showQuad && preloadHit ? (
        <div className="lm-cable-preload-pool" aria-hidden="true">
          <CableStreamSlot
            key={`preload-${preloadHit.id}-${preloadFavIdx}`}
            hit={preloadHit}
            globalSnow={false}
            osdVisible={false}
            channelNum={0}
            preload
            loadTimeoutMs={CABLE_STREAM_LOAD_MS}
            onStreamReady={() => {
              if (preloadFavIdx === warmSwitchTarget) setPreloadReady(true);
            }}
            onStreamFail={advancePreloadCandidate}
          />
        </div>
      ) : null}
    </div>
  );
}
