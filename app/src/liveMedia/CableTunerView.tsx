import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "../searchResults/types";
import { fetchStartupContext, getStartupContextSync, refreshLocalWeather } from "../startupContext";
import type { StartupContext } from "../startupContext/types";
import { CableStreamSlot } from "./CableStreamSlot";
import { CableRadioHeaderStrip } from "./CableRadioHeaderStrip";
import { ClassicRadioView } from "./ClassicRadioView";
import { formatCableOsdDate, formatCableOsdWeather, shortenCableChannelTitle } from "./cableOsdContext";
import {
  RADIO_INTERSTITIAL_EVERY,
  firstRadioCablePage,
  isRadioCablePage,
  maxCablePageIndexTvRadio,
  nextCablePageWithRadio,
  radioCablePageIndex,
} from "./cableTunerRadio";
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
  nextFavoriteIndex,
  nextWorkingFavoriteIndex,
  pageIndexForFavorite,
  pickCableQuadFromSlots,
  prevFavoriteIndex,
  singleFavoriteIndex,
  targetFavoriteAfterStep,
} from "./cableTunerUtils";
import { loadCableTunerSession, saveCableTunerSession } from "./cableTunerSession";
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
  /** Regional radio lineup (geo + favorites) — interstitials + classic radio pages. */
  regionalRadio?: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  loading: boolean;
  onOpenBrowse: () => void;
  onRemoveFavorite: (hit: UnifiedSearchHit) => void | Promise<void>;
  /** Mobile / side drawer: show pinned back control to leave TV mode */
  onBack?: () => void;
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

export function CableTunerView({
  favorites,
  regionalRadio = [],
  uiLang,
  loading,
  onOpenBrowse,
  onRemoveFavorite,
  onBack,
}: Props) {
  const rtl = uiLang === "he";
  const [pageIndex, setPageIndex] = useState(0);
  const [quadSlots, setQuadSlots] = useState<number[]>(() => initialQuadSlots(favorites.length));
  const [rotationCursor, setRotationCursor] = useState(() =>
    initialRotationCursor(favorites.length, initialQuadSlots(favorites.length)),
  );
  const [selectedQuadSlot, setSelectedQuadSlot] = useState(0);
  const [quadActionSlot, setQuadActionSlot] = useState<number | null>(null);
  const rotationSlotRef = useRef(0);
  const quadStateRef = useRef({
    slots: initialQuadSlots(favorites.length),
    cursor: initialRotationCursor(favorites.length, initialQuadSlots(favorites.length)),
  });
  const [globalSnow, setGlobalSnow] = useState(false);
  const [osdVisible, setOsdVisible] = useState(true);
  const [confirmRemoveOpen, setConfirmRemoveOpen] = useState(false);
  const [channelMenuOpen, setChannelMenuOpen] = useState(false);
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
  const sessionHydratedRef = useRef(false);

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
          jumpToChannel: (n: number) => `▶ לערוץ ${n}`,
          backQuad: "מסך מפוצל",
          back: "חזרה",
          more: "אפשרויות ערוץ",
          radioBand: "רדיו",
          openRadio: "פתח רדיו",
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
          jumpToChannel: (n: number) => `▶ CH ${n}`,
          backQuad: "Split screen",
          back: "Back",
          more: "Channel options",
          radioBand: "Radio",
          openRadio: "Open radio",
        };

  const total = favorites.length;
  const radioTotal = regionalRadio.length;
  const showQuad = isQuadPage(pageIndex) && !isRadioCablePage(pageIndex, total);
  const showRadioPage = isRadioCablePage(pageIndex, total);
  const radioIndex = showRadioPage ? radioCablePageIndex(pageIndex, total) : -1;
  const radioHit = showRadioPage ? (regionalRadio[radioIndex] ?? null) : null;
  const [headerRadioIdx, setHeaderRadioIdx] = useState(0);
  const [headerRadioAnim, setHeaderRadioAnim] = useState<"in" | "out">("in");
  const [headerRadioAudio, setHeaderRadioAudio] = useState(false);
  const headerRadioTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const radioRotateRef = useRef(0);
  const quadTiles = useMemo(() => pickCableQuadFromSlots(favorites, quadSlots), [favorites, quadSlots]);
  const singleHit = useMemo(() => favoriteForPage(favorites, pageIndex), [favorites, pageIndex]);
  const focusHit = showRadioPage ? radioHit : showQuad ? (quadTiles[selectedQuadSlot] ?? null) : singleHit;

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

  const range = showRadioPage
    ? { from: radioIndex + 1, to: radioIndex + 1 }
    : cableOsdRangeLabel(pageIndex, total, showQuad ? quadSlots : undefined);
  const rangeLabel = showRadioPage
    ? `${L.radioBand} ${radioIndex + 1}${radioTotal > 1 ? `/${radioTotal}` : ""}`
    : range
      ? L.range(range.from, range.to)
      : "";
  const regionLabel = localCtx?.cityName ?? localCtx?.countryName ?? localCtx?.countryCode?.toUpperCase() ?? "—";
  const dateLabel = formatCableOsdDate(now, uiLang, localCtx?.timezone);
  const weatherLabel = formatCableOsdWeather(localCtx);
  const clock = formatClock(now, rtl, localCtx?.timezone);
  const centerTitle = focusHit ? shortenCableChannelTitle(focusHit.title) : "";

  const bumpHeaderRadio = useCallback(
    (nextIdx: number) => {
      if (regionalRadio.length < 1) return;
      const idx = ((nextIdx % regionalRadio.length) + regionalRadio.length) % regionalRadio.length;
      if (headerRadioTimer.current) clearTimeout(headerRadioTimer.current);
      setHeaderRadioAnim("out");
      headerRadioTimer.current = setTimeout(() => {
        setHeaderRadioIdx(idx);
        setHeaderRadioAnim("in");
        setHeaderRadioAudio(false);
        headerRadioTimer.current = null;
      }, 340);
    },
    [regionalRadio.length],
  );

  const pokeOsd = useCallback(() => {
    setOsdVisible(true);
    if (osdTimer.current) clearTimeout(osdTimer.current);
    if (showRadioPage) return;
    if (!confirmRemoveOpen && !channelMenuOpen) {
      osdTimer.current = setTimeout(() => setOsdVisible(false), OSD_HIDE_MS);
    }
  }, [channelMenuOpen, confirmRemoveOpen, showRadioPage]);

  const closeConfirmRemove = useCallback(() => {
    setConfirmRemoveOpen(false);
    setChannelMenuOpen(false);
    pokeOsd();
  }, [pokeOsd]);

  const openConfirmRemove = useCallback(() => {
    if (!focusHit) return;
    if (osdTimer.current) clearTimeout(osdTimer.current);
    setChannelMenuOpen(false);
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

  const goToPage = useCallback(
    (targetPage: number, opts?: { tuneMs?: number }) => {
      const maxPage = maxCablePageIndexTvRadio(total, radioTotal);
      if (maxPage < 1 && total < 1) return;
      if (globalSnow) return;
      if (targetPage === pageIndex) return;
      clearTuneTimer();
      pokeOsd();

      const onRadioTarget = isRadioCablePage(targetPage, total);
      const onRadioSource = isRadioCablePage(pageIndex, total);
      if (onRadioSource && onRadioTarget) {
        setPageIndex(targetPage);
        setAudioUnlocked(true);
        return;
      }

      const targetFav =
        isQuadPage(targetPage) || onRadioTarget ? -1 : singleFavoriteIndex(targetPage);
      const warmHit =
        opts?.tuneMs == null &&
        !isQuadPage(targetPage) &&
        !onRadioTarget &&
        preloadReady &&
        targetFav >= 0 &&
        preloadFavIdx === targetFav;
      const tuneMs = opts?.tuneMs ?? (warmHit ? CABLE_WARM_SWITCH_MS : TUNE_MS);

      setGlobalSnow(true);
      tuneTimer.current = setTimeout(() => {
        setPageIndex(targetPage);
        setGlobalSnow(false);
        setQuadActionSlot(null);
        if (warmHit) setPreloadReady(false);
      }, tuneMs);
    },
    [clearTuneTimer, globalSnow, pageIndex, pokeOsd, preloadFavIdx, preloadReady, radioTotal, total],
  );

  const goToRadioFull = useCallback(
    (radioIdx: number) => {
      if (radioTotal < 1) return;
      const idx = ((radioIdx % radioTotal) + radioTotal) % radioTotal;
      goToPage(firstRadioCablePage(total) + idx, { tuneMs: CABLE_WARM_SWITCH_MS });
    },
    [goToPage, radioTotal, total],
  );

  const jumpToFavoriteFromQuad = useCallback(
    (favoriteIndex: number) => {
      if (favoriteIndex < 0 || favoriteIndex >= total) return;
      goToPage(pageIndexForFavorite(favoriteIndex), { tuneMs: CABLE_WARM_SWITCH_MS });
    },
    [goToPage, total],
  );

  const goToQuad = useCallback(() => {
    goToPage(0, { tuneMs: CABLE_WARM_SWITCH_MS });
  }, [goToPage]);

  const changePage = useCallback(
    (delta: 1 | -1) => {
      if (globalSnow) return;
      const maxPage = maxCablePageIndexTvRadio(total, radioTotal);
      if (maxPage < 1) return;

      if (isRadioCablePage(pageIndex, total)) {
        if (radioTotal <= 1) return;
        const first = firstRadioCablePage(total);
        const last = first + radioTotal - 1;
        let next = pageIndex + delta;
        if (next > last) next = first;
        if (next < first) next = last;
        goToPage(next);
        return;
      }

      if (radioTotal > 0) {
        if (delta === 1 && pageIndex === total) {
          goToPage(firstRadioCablePage(total));
          return;
        }
        if (delta === -1 && pageIndex === 0) {
          goToPage(maxPage);
          return;
        }
      }

      if (total < 2) {
        if (radioTotal > 0) {
          goToPage(nextCablePageWithRadio(pageIndex, delta, total, radioTotal));
        }
        return;
      }

      let targetPage: number;
      if (isQuadPage(pageIndex)) {
        const targetFav = targetFavoriteAfterStep(pageIndex, delta, total, deadFavorites);
        targetPage = pageIndexForFavorite(targetFav);
      } else {
        const curFav = singleFavoriteIndex(pageIndex);
        if (delta === -1 && curFav === 0) {
          targetPage = 0;
        } else if (delta === 1 && pageIndex === total) {
          targetPage = radioTotal > 0 ? firstRadioCablePage(total) : 0;
        } else {
          const targetFav = nextWorkingFavoriteIndex(curFav, delta, total, deadFavorites);
          targetPage = pageIndexForFavorite(targetFav);
        }
      }

      goToPage(targetPage);
    },
    [deadFavorites, globalSnow, goToPage, pageIndex, radioTotal, total],
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
    if (total <= 0) return;
    if (!sessionHydratedRef.current) {
      sessionHydratedRef.current = true;
      const saved = loadCableTunerSession();
      if (saved) {
        const slots = saved.quadSlots.map((i) => i % total);
        const cursor = saved.rotationCursor % total;
        quadStateRef.current = { slots, cursor };
        setQuadSlots(slots);
        setRotationCursor(cursor);
        setPageIndex(Math.min(saved.pageIndex, maxCablePageIndexTvRadio(total, radioTotal)));
        setSelectedQuadSlot(saved.selectedQuadSlot);
        rotationSlotRef.current = 0;
        return;
      }
      const slots = initialQuadSlots(total);
      const cursor = initialRotationCursor(total, slots);
      quadStateRef.current = { slots, cursor };
      setQuadSlots(slots);
      setRotationCursor(cursor);
      rotationSlotRef.current = 0;
      setSelectedQuadSlot(0);
      return;
    }
    setPageIndex((p) => Math.min(p, maxCablePageIndexTvRadio(total, radioTotal)));
    setQuadSlots((slots) => slots.map((i) => i % total));
    setRotationCursor((c) => c % total);
  }, [total, radioTotal]);

  useEffect(() => {
    if (total <= 0) return;
    saveCableTunerSession({ pageIndex, quadSlots, rotationCursor, selectedQuadSlot });
  }, [pageIndex, quadSlots, rotationCursor, selectedQuadSlot, total]);

  useEffect(() => {
    return () => {
      if (total <= 0) return;
      saveCableTunerSession({ pageIndex, quadSlots, rotationCursor, selectedQuadSlot });
    };
  }, [pageIndex, quadSlots, rotationCursor, selectedQuadSlot, total]);

  useEffect(() => {
    setDeadFavorites(new Set());
  }, [favorites]);

  useEffect(() => {
    if (showQuad || total < 2) return;
    setPreloadFavIdx(targetFavoriteAfterStep(pageIndex, 1, total, deadFavorites));
    setPreloadReady(false);
  }, [deadFavorites, pageIndex, showQuad, total]);

  useEffect(() => {
    if (!showQuad || total < 2) return;
    const id = window.setInterval(() => {
      const slot = rotationSlotRef.current % 4;
      rotationSlotRef.current += 1;
      const advanced = advanceQuadRotation(quadStateRef.current.slots, slot, quadStateRef.current.cursor, total);
      quadStateRef.current = advanced;
      setQuadSlots(advanced.slots);
      setRotationCursor(advanced.cursor);

      radioRotateRef.current += 1;
      if (regionalRadio.length > 0 && radioRotateRef.current % RADIO_INTERSTITIAL_EVERY === 0) {
        const radioIdx =
          Math.floor(radioRotateRef.current / RADIO_INTERSTITIAL_EVERY) % regionalRadio.length;
        bumpHeaderRadio(radioIdx);
      }

      pokeOsd();
    }, QUAD_ROTATE_MS);
    return () => window.clearInterval(id);
  }, [bumpHeaderRadio, showQuad, total, regionalRadio, pokeOsd]);

  useEffect(() => {
    if (!showQuad || !osdVisible || regionalRadio.length < 2) return;
    const id = window.setInterval(() => {
      bumpHeaderRadio(headerRadioIdx + 1);
    }, 4500);
    return () => window.clearInterval(id);
  }, [bumpHeaderRadio, headerRadioIdx, osdVisible, regionalRadio.length, showQuad]);

  useEffect(() => {
    if (!showQuad) {
      setHeaderRadioAudio(false);
      setHeaderRadioAnim("in");
    }
  }, [showQuad]);

  useEffect(
    () => () => {
      if (headerRadioTimer.current) clearTimeout(headerRadioTimer.current);
    },
    [],
  );

  useEffect(() => {
    if (!osdVisible) {
      setQuadActionSlot(null);
      setHeaderRadioAudio(false);
    }
  }, [osdVisible]);

  useEffect(() => {
    if (!osdVisible) setChannelMenuOpen(false);
  }, [osdVisible]);

  useEffect(() => {
    if (!channelMenuOpen) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null;
      if (t?.closest(".lm-cable-channel-menu")) return;
      setChannelMenuOpen(false);
      if (osdTimer.current) clearTimeout(osdTimer.current);
      if (!confirmRemoveOpen && !showRadioPage) {
        osdTimer.current = setTimeout(() => setOsdVisible(false), OSD_HIDE_MS);
      }
    };
    document.addEventListener("click", onDoc);
    return () => document.removeEventListener("click", onDoc);
  }, [channelMenuOpen, confirmRemoveOpen, showRadioPage]);

  useEffect(() => {
    if (!showQuad) setQuadActionSlot(null);
  }, [showQuad]);

  useEffect(() => {
    if (!showRadioPage) return;
    setOsdVisible(true);
    if (osdTimer.current) clearTimeout(osdTimer.current);
  }, [showRadioPage]);

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
      if (channelMenuOpen && e.key === "Escape") {
        e.preventDefault();
        setChannelMenuOpen(false);
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
      if (!showQuad && (e.key === "0" || e.key === "Home")) {
        e.preventDefault();
        goToQuad();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [changePage, channelMenuOpen, closeConfirmRemove, confirmRemoveOpen, goToQuad, pokeOsd, showQuad, toggleFullscreen]);

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
      {showRadioPage && radioHit ? (
        <div className="lm-cable-single lm-cable-single--radio">
          <ClassicRadioView
            hit={radioHit}
            stationIndex={radioIndex}
            stationTotal={radioTotal}
            regionLabel={regionLabel}
            uiLang={uiLang}
            muted={slotMuted(true)}
            volume={volume}
          />
        </div>
      ) : showQuad ? (
        <div className="lm-cable-grid lm-cable-grid--4">
          {Array.from({ length: 4 }, (_, i) => {
            const hit = quadTiles[i] ?? null;
            const audioFocus = selectedQuadSlot === i;
            const chNum = quadSlots[i] != null ? quadSlots[i] + 1 : 0;

            return (
              <CableStreamSlot
                key={`quad-slot-${i}`}
                hit={hit}
                globalSnow={globalSnow}
                osdVisible={osdVisible}
                channelNum={chNum}
                channelBadgeTopRight
                selected={audioFocus}
                audioFocus={audioFocus && audioUnlocked && !userMuted && !headerRadioAudio}
                muted={slotMuted(audioFocus && !headerRadioAudio)}
                volume={volume}
                multiView
                loadTimeoutMs={CABLE_STREAM_LOAD_MS}
                quadJumpOpen={quadActionSlot === i && osdVisible && !globalSnow}
                quadJumpLabel={chNum > 0 ? L.jumpToChannel(chNum) : ""}
                onQuadJump={() => jumpToFavoriteFromQuad(quadSlots[i] ?? 0)}
                onDoubleActivate={() => jumpToFavoriteFromQuad(quadSlots[i] ?? 0)}
                onStreamFail={() => handleQuadSlotFail(i)}
                onSelect={() => {
                  setSelectedQuadSlot(i);
                  setAudioUnlocked(true);
                  setHeaderRadioAudio(false);
                  setQuadActionSlot((prev) => (prev === i ? null : i));
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
            channelBadgeTopRight
            audioFocus={audioUnlocked && !userMuted}
            muted={slotMuted(true)}
            volume={volume}
            single
            loadTimeoutMs={CABLE_STREAM_LOAD_MS}
            onStreamFail={skipSingleToNext}
          />
        </div>
      ) : null}

      {onBack || !showQuad || (showQuad && radioTotal > 0) ? (
        <header
          className={`lm-cable-header${osdVisible ? "" : " is-hidden"}${showQuad && radioTotal > 0 ? " lm-cable-header--quad-radio" : ""}`}
          dir={rtl ? "rtl" : "ltr"}
        >
          <div
            className={`lm-cable-header-inner${showQuad && radioTotal > 0 ? " lm-cable-header-inner--quad-radio" : ""}`}
          >
            {showQuad && radioTotal > 0 ? (
              <CableRadioHeaderStrip
                stations={regionalRadio}
                activeIndex={headerRadioIdx}
                animPhase={headerRadioAnim}
                uiLang={uiLang}
                audioFocus={headerRadioAudio && audioUnlocked && !userMuted}
                muted={slotMuted(headerRadioAudio)}
                volume={volume}
                onActivate={() => {
                  setHeaderRadioAudio(true);
                  setAudioUnlocked(true);
                  pokeOsd();
                }}
                onOpenFull={(idx) => goToRadioFull(idx)}
                onSelectIndex={bumpHeaderRadio}
              />
            ) : (
              <>
                {onBack ? (
                  <button
                    type="button"
                    className="lm-cable-osd-btn lm-cable-osd-btn--back"
                    onClick={onBack}
                    aria-label={L.back}
                    title={L.back}
                  >
                    <span className="lm-cable-osd-btn-glyph" aria-hidden="true">
                      {rtl ? "→" : "←"}
                    </span>
                    <span>{L.back}</span>
                  </button>
                ) : (
                  <span className="lm-cable-header-slot" aria-hidden="true" />
                )}

                <div className="lm-cable-channel-menu">
                  <button
                    type="button"
                    className="lm-cable-osd-btn lm-cable-osd-btn--icon lm-cable-osd-btn--menu"
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = !channelMenuOpen;
                      setChannelMenuOpen(next);
                      setOsdVisible(true);
                      if (osdTimer.current) clearTimeout(osdTimer.current);
                      if (!confirmRemoveOpen && !next) {
                        osdTimer.current = setTimeout(() => setOsdVisible(false), OSD_HIDE_MS);
                      }
                    }}
                    disabled={!focusHit}
                    aria-label={L.more}
                    aria-expanded={channelMenuOpen}
                    title={L.more}
                  >
                    ⋮
                  </button>
                  {channelMenuOpen && focusHit ? (
                    <div className="lm-cable-channel-menu-pop" role="menu">
                      <button
                        type="button"
                        className="lm-cable-channel-menu-item lm-cable-channel-menu-item--danger"
                        role="menuitem"
                        onClick={openConfirmRemove}
                      >
                        {L.removeFav}
                      </button>
                    </div>
                  ) : null}
                </div>
              </>
            )}
          </div>
        </header>
      ) : null}

      {confirmRemoveOpen && focusHit ? (
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

      <div
        className={`lm-cable-osd${showRadioPage ? " lm-cable-osd--pinned" : osdVisible ? "" : " is-hidden"}`}
        dir="ltr"
      >
        <div className="lm-cable-osd-actions">
          <button
            type="button"
            className="lm-cable-osd-btn lm-cable-osd-btn--remote"
            onClick={() => changePage(1)}
            disabled={globalSnow || (showRadioPage ? radioTotal <= 1 : total <= 1 && radioTotal === 0)}
            aria-label={L.chUp}
          >
            {L.chUp}
          </button>
          <button
            type="button"
            className="lm-cable-osd-btn lm-cable-osd-btn--remote"
            onClick={() => changePage(-1)}
            disabled={globalSnow || (showRadioPage ? radioTotal <= 1 : total <= 1 && radioTotal === 0)}
            aria-label={L.chDown}
          >
            {L.chDown}
          </button>
          {!showQuad ? (
            <button
              type="button"
              className="lm-cable-osd-btn lm-cable-osd-btn--quad"
              onClick={goToQuad}
              disabled={globalSnow}
              aria-label={L.backQuad}
              title={L.backQuad}
            >
              ⊞
            </button>
          ) : null}
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
            <div className="lm-cable-volume-rail">
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
                onPointerDown={() => pokeOsd()}
                aria-label={L.volume}
              />
            </div>
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

      {!showQuad && !showRadioPage && preloadHit ? (
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
