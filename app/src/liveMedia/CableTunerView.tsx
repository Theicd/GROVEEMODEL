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
  shiftQuadLineup,
  singleFavoriteIndex,
  targetFavoriteAfterStep,
} from "./cableTunerUtils";
import { loadCableTunerSession, saveCableTunerSession } from "./cableTunerSession";
import { CableEpgPanel } from "./CableEpgPanel";
import { useEpgGuide } from "./epg/useEpgGuide";
import { warmMjhEpgCaches } from "./epg/epgService";
import { useStreamCueSync } from "./epg/useStreamCueSync";
import { useTmdbProgramMeta } from "./epg/useTmdbProgramMeta";
import { useLocalizedEpgNowPlaying } from "./epg/useLocalizedEpgNowPlaying";
import { resolveLiveEpgProgram } from "./epg/epgProgramSync";
import {
  entryForHit,
  formatDurationMinutes,
  formatEpisodeLabel,
  formatOsdProgramRange,
  nowPlayingFromEntry,
} from "./epg/epgNowPlaying";
import {
  CableTunerWelcome,
  dismissTvWelcome,
  readTvWelcomeDismissed,
} from "./CableTunerWelcome";
import { CableTunerGearMenu } from "./CableTunerGearMenu";
import { LiveCaptionsOverlay } from "./LiveCaptionsOverlay";
import { broadcastLangToSpeechCode, CAPTION_TARGET_NONE } from "./liveTranslate";
import { isLiveCaptionsSupported, useLiveCaptions } from "./useLiveCaptions";
import type { UserChannelCategory } from "./channelUserTaxonomy";
import "./cableTuner.css";
const OSD_HIDE_MS = 4500;
const VOLUME_KEY = "grovee-cable-volume";
/** Analog-style tuning duration (snow + antenna scan meter) when switching channels. */
const TUNE_MS = 1500;

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
  /** Category group order for TV guide sections. */
  categoryOrder?: UserChannelCategory[];
  /** Regional radio lineup (geo + favorites) — interstitials + classic radio pages. */
  regionalRadio?: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  loading: boolean;
  onOpenBrowse: () => void;
  onRemoveFavorite: (hit: UnifiedSearchHit) => void | Promise<void>;
  /** Mobile / side drawer: show pinned back control to leave TV mode */
  onBack?: () => void;
  /**
   * "supersport" = distributable sports-only cable experience: branded boot splash,
   * landscape-first split screen, edge channel-scroll rails, and pulsing full-screen prompt.
   */
  profile?: "default" | "supersport";
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
  categoryOrder,
  regionalRadio = [],
  uiLang,
  loading,
  onOpenBrowse,
  onRemoveFavorite,
  onBack,
  profile = "default",
}: Props) {
  const rtl = uiLang === "he";
  const superSport = profile === "supersport";
  const [pageIndex, setPageIndex] = useState(0);
  const [quadSlots, setQuadSlots] = useState<number[]>(() => initialQuadSlots(favorites.length));
  const [rotationCursor, setRotationCursor] = useState(() =>
    initialRotationCursor(favorites.length, initialQuadSlots(favorites.length)),
  );
  const [selectedQuadSlot, setSelectedQuadSlot] = useState(0);
  const rotationSlotRef = useRef(0);
  const quadStateRef = useRef({
    slots: initialQuadSlots(favorites.length),
    cursor: initialRotationCursor(favorites.length, initialQuadSlots(favorites.length)),
  });
  const [globalSnow, setGlobalSnow] = useState(false);
  // SUPER SPORT skips the multi-step tour and shows a short branded boot splash instead.
  const [welcomeOpen, setWelcomeOpen] = useState(() => (profile === "supersport" ? false : !readTvWelcomeDismissed()));
  const [bootSplash, setBootSplash] = useState(() => profile === "supersport");
  const [osdVisible, setOsdVisible] = useState(true);
  const [confirmRemoveOpen, setConfirmRemoveOpen] = useState(false);
  const [channelMenuOpen, setChannelMenuOpen] = useState(false);
  const [gearMenuOpen, setGearMenuOpen] = useState(false);
  const [volumeOpen, setVolumeOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [localCtx, setLocalCtx] = useState<StartupContext | null>(() => getStartupContextSync());
  const [now, setNow] = useState(() => new Date());
  const [volume, setVolume] = useState(readStoredVolume);
  const [userMuted, setUserMuted] = useState(false);
  const [audioUnlocked, setAudioUnlocked] = useState(false);
  const [preloadFavIdx, setPreloadFavIdx] = useState(0);
  const [preloadReady, setPreloadReady] = useState(false);
  const [deadFavorites, setDeadFavorites] = useState<ReadonlySet<number>>(() => new Set());
  const [epgOpen, setEpgOpen] = useState(false);
  const tuneTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const osdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const singleVideoRef = useRef<HTMLVideoElement | null>(null);
  const gearBtnRef = useRef<HTMLButtonElement | null>(null);
  const sessionHydratedRef = useRef(false);
  const quadStreamReadyRef = useRef(new Set<string>());
  const [assumeSingleReady, setAssumeSingleReady] = useState(false);
  const captions = useLiveCaptions();
  const {
    status: captionsStatus,
    statusMessage: captionsStatusMessage,
    loadPct: captionsLoadPct,
    original: captionsOriginal,
    translated: captionsTranslated,
    active: captionsActive,
    start: startCaptions,
    stop: stopCaptions,
  } = captions;
  const [captionSourceLang, setCaptionSourceLang] = useState(() =>
    broadcastLangToSpeechCode(uiLang === "he" ? "eng" : "heb"),
  );
  const [captionTargetLang, setCaptionTargetLang] = useState(CAPTION_TARGET_NONE);
  const captionsSupported = isLiveCaptionsSupported();

  const L =
    uiLang === "he"
      ? {
          noFavs: "אין ערוצים במועדפים — פתח חיפוש והוסף ☆ לערוצים שעובדים.",
          browse: "חיפוש",
          chUp: "▲",
          chDown: "▼",
          chUpLbl: "למעלה",
          chDownLbl: "למטה",
          menuLbl: "מפוצל 4",
          settingsLbl: "הגדרות",
          fullscreenLbl: "מסך מלא",
          volumeLbl: "ווליום",
          quadScreen: "מסך מפוצל",
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
          quadHint: "▲▼ החלפת ערוצים · ◀▶ בחירת שמע · ▶ מסך מלא",
          quadAudioOn: (t: string) => `שמע: ${t}`,
          quadPickChannel: "בחר ערוץ",
          backQuad: "מסך מפוצל",
          back: "חזרה",
          more: "אפשרויות ערוץ",
          radioBand: "רדיו",
          openRadio: "פתח רדיו",
          epg: "לוח שידורים",
          onNow: "עכשיו",
          endsIn: (m: number) => (m <= 0 ? "מסתיים" : `נותרו ${m} דק`),
        }
      : {
          noFavs: "No favorites — open search and star ☆ working channels.",
          browse: "Search",
          chUp: "▲",
          chDown: "▼",
          chUpLbl: "Up",
          chDownLbl: "Down",
          menuLbl: "Split 4",
          settingsLbl: "Settings",
          fullscreenLbl: "Fullscreen",
          volumeLbl: "Volume",
          quadScreen: "Multi view",
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
          quadHint: "▲▼ shift channels · ◀▶ pick audio · ▶ full screen",
          quadAudioOn: (t: string) => `Audio: ${t}`,
          quadPickChannel: "Pick a channel",
          backQuad: "Split screen",
          back: "Back",
          more: "Channel options",
          radioBand: "Radio",
          openRadio: "Open radio",
          epg: "TV guide",
          onNow: "On now",
          endsIn: (m: number) => (m <= 0 ? "ending" : `${m} min left`),
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
  const epgHit = showRadioPage || showQuad ? null : focusHit;
  const streamUrl = focusHit?.mediaPlayUrl || focusHit?.url;
  const streamCue = useStreamCueSync(streamUrl, Boolean(focusHit) && !showQuad && !showRadioPage && !globalSnow);
  const epgGuide = useEpgGuide(favorites, Boolean(favorites.length) && !showRadioPage && !showQuad);

  const focusChannelId = focusHit?.id ?? null;
  const epgEntry = useMemo(
    () => (focusChannelId ? entryForHit(epgGuide.entries, focusChannelId) : null),
    [epgGuide.entries, focusChannelId],
  );
  const epgLiveProgram = useMemo(() => {
    const programs = epgEntry?.schedule?.programs;
    if (!programs?.length) return null;
    return resolveLiveEpgProgram(programs, now, {
      cue: streamCue,
      streamUrl,
      sourceKey: epgEntry?.schedule?.channel?.sourceKey,
      tvgId: typeof focusHit?.meta?.tvgId === "string" ? focusHit.meta.tvgId : undefined,
    });
  }, [epgEntry, now, streamCue, streamUrl]);
  const tmdbMeta = useTmdbProgramMeta(
    epgLiveProgram,
    Boolean(epgLiveProgram?.title) && Boolean(focusChannelId),
    uiLang,
    focusHit?.title,
    focusChannelId,
  );

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
  const regionLabel = localCtx?.cityName ?? localCtx?.countryName ?? localCtx?.countryCode?.toUpperCase() ?? "—";
  const dateLabel = formatCableOsdDate(now, uiLang, localCtx?.timezone);
  const weatherLabel = formatCableOsdWeather(localCtx);
  const clock = formatClock(now, rtl, localCtx?.timezone);
  const centerTitle = focusHit ? shortenCableChannelTitle(focusHit.title) : "";
  const osdChannelNum = showRadioPage
    ? radioIndex + 1
    : showQuad
      ? null
      : singleFavoriteIndex(pageIndex) + 1;
  const osdChannelRange = showQuad && range ? `${range.from}${range.to !== range.from ? `–${range.to}` : ""}` : null;
  const nowPlayingInfo = useMemo(() => {
    if (!focusHit || !focusChannelId || showQuad || showRadioPage) return null;
    if (epgEntry && epgEntry.hit.id !== focusChannelId) return null;
    return nowPlayingFromEntry(epgEntry, now, streamCue, tmdbMeta?.runtimeMinutes ?? null);
  }, [epgEntry, focusChannelId, focusHit, now, showQuad, showRadioPage, streamCue, tmdbMeta?.runtimeMinutes]);

  const tmdbOverviewShort =
    tmdbMeta?.overview && tmdbMeta.overview.length > 0 ?
      tmdbMeta.overview.length > 140 ?
        `${tmdbMeta.overview.slice(0, 139).trim()}…`
      : tmdbMeta.overview
    : null;

  const nowPlayingLocalized = useLocalizedEpgNowPlaying(
    nowPlayingInfo,
    uiLang,
    tmdbMeta?.title,
    tmdbOverviewShort,
    tmdbMeta?.seriesTitle,
    focusChannelId,
  );

  const nowPlaying = useMemo(() => {
    if (!nowPlayingLocalized || !nowPlayingInfo || !focusChannelId) return null;
    if (epgEntry && epgEntry.hit.id !== focusChannelId) return null;
    const description = nowPlayingLocalized.description;
    const clippedDesc =
      description && description.length > 140 ? `${description.slice(0, 139).trim()}…` : description;
    return {
      ...nowPlayingInfo,
      displayTitle: nowPlayingLocalized.displayTitle,
      seriesTitle: nowPlayingLocalized.seriesTitle,
      episodeLabel: formatEpisodeLabel(nowPlayingInfo.program),
      description: clippedDesc,
      tmdbYear: tmdbMeta?.year ?? null,
      tmdbRating: tmdbMeta?.rating ?? null,
      posterUrl: tmdbMeta?.posterUrl ?? null,
    };
  }, [nowPlayingInfo, nowPlayingLocalized, tmdbMeta?.year, tmdbMeta?.rating, tmdbMeta?.posterUrl, epgEntry, focusChannelId]);

  const markQuadStreamReady = useCallback((hitId: string) => {
    quadStreamReadyRef.current.add(hitId);
  }, []);

  const osdNoProgram = !showQuad && !showRadioPage && !nowPlaying;
  const singleChannelView = !showQuad && !showRadioPage && Boolean(singleHit);

  const toggleCaptions = useCallback(async () => {
    if (captionsActive) {
      stopCaptions();
      return;
    }
    setAudioUnlocked(true);
    setUserMuted(false);
    const v = singleVideoRef.current;
    if (v) {
      v.muted = false;
      v.volume = Math.max(v.volume, volume > 0 ? volume : 0.75);
      try {
        await v.play();
      } catch {
        /* ignore */
      }
    }
    const bl = (focusHit?.meta?.broadcastLanguage as string | undefined) ?? "eng";
    const src = captionSourceLang || broadcastLangToSpeechCode(bl);
    void startCaptions({
      video: v,
      sourceLang: src,
      targetLang: captionTargetLang,
    });
  }, [
    captionSourceLang,
    captionTargetLang,
    captionsActive,
    focusHit?.meta?.broadcastLanguage,
    startCaptions,
    stopCaptions,
    volume,
  ]);

  useEffect(() => {
    stopCaptions();
    if (!singleChannelView) return;
    const bl = (focusHit?.meta?.broadcastLanguage as string | undefined) ?? "eng";
    setCaptionSourceLang(broadcastLangToSpeechCode(bl));
  }, [pageIndex, singleChannelView, focusHit?.id, focusHit?.meta?.broadcastLanguage, stopCaptions]);

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
    if (!confirmRemoveOpen && !channelMenuOpen && !gearMenuOpen && !volumeOpen) {
      osdTimer.current = setTimeout(() => setOsdVisible(false), OSD_HIDE_MS);
    }
  }, [channelMenuOpen, confirmRemoveOpen, gearMenuOpen, showRadioPage, volumeOpen]);

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
    if (showQuad || total < 2) return;
    const cur = singleFavoriteIndex(pageIndex);
    markFavoriteDead(cur);
    const nextFav = nextWorkingFavoriteIndex(cur, 1, total, new Set([...deadFavorites, cur]));
    const targetPage = pageIndexForFavorite(nextFav);
    clearTuneTimer();
    setPageIndex(targetPage);
    setGlobalSnow(true);
    tuneTimer.current = setTimeout(() => {
      setGlobalSnow(false);
    }, TUNE_MS);
  }, [clearTuneTimer, deadFavorites, markFavoriteDead, pageIndex, showQuad, total]);

  const goToPage = useCallback(
    (targetPage: number, opts?: { tuneMs?: number; assumeStreamReady?: boolean }) => {
      const maxPage = maxCablePageIndexTvRadio(total, radioTotal);
      if (maxPage < 1 && total < 1) return;
      if (globalSnow) return;
      if (targetPage === pageIndex) return;
      clearTuneTimer();
      pokeOsd();
      setAssumeSingleReady(Boolean(opts?.assumeStreamReady));

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

      if (isQuadPage(targetPage)) {
        setAssumeSingleReady(false);
      }

      if (tuneMs <= 0) {
        setPageIndex(targetPage);
        setGlobalSnow(false);
        if (warmHit) setPreloadReady(false);
        return;
      }

      setGlobalSnow(true);
      tuneTimer.current = setTimeout(() => {
        setPageIndex(targetPage);
        setGlobalSnow(false);
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
      const hit = favorites[favoriteIndex];
      const ready = Boolean(hit && quadStreamReadyRef.current.has(hit.id));
      goToPage(pageIndexForFavorite(favoriteIndex), {
        tuneMs: ready ? 0 : CABLE_WARM_SWITCH_MS,
        assumeStreamReady: ready,
      });
    },
    [favorites, goToPage, total],
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

      if (radioTotal > 0 && delta === 1 && pageIndex === total) {
        goToPage(firstRadioCablePage(total));
        return;
      }

      if (total < 2) {
        if (isQuadPage(pageIndex) && total === 1) {
          goToPage(1);
          return;
        }
        if (radioTotal > 0) {
          goToPage(nextCablePageWithRadio(pageIndex, delta, total, radioTotal));
        }
        return;
      }

      if (isQuadPage(pageIndex)) {
        if (total <= 1) {
          if (total === 1) goToPage(1);
          return;
        }
        const shifted = shiftQuadLineup(quadStateRef.current.slots, delta, total, deadFavorites);
        quadStateRef.current = { slots: shifted.slots, cursor: shifted.cursor };
        setQuadSlots(shifted.slots);
        setRotationCursor(shifted.cursor);
        pokeOsd();
        return;
      }

      let targetPage: number;
      const curFav = singleFavoriteIndex(pageIndex);
      if (delta === -1 && curFav === 0) {
        targetPage = 0;
      } else if (delta === 1 && pageIndex === total) {
        targetPage = radioTotal > 0 ? firstRadioCablePage(total) : 0;
      } else {
        const targetFav = nextWorkingFavoriteIndex(curFav, delta, total, deadFavorites);
        targetPage = pageIndexForFavorite(targetFav);
      }

      goToPage(targetPage);
    },
    [deadFavorites, globalSnow, goToPage, pageIndex, pokeOsd, radioTotal, total],
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
    void warmMjhEpgCaches();
  }, []);

  // SUPER SPORT: dismiss the branded boot splash once channels are ready (min ~2.2s of "connecting").
  useEffect(() => {
    if (!superSport || !bootSplash) return;
    if (loading || total < 1) return;
    const t = setTimeout(() => setBootSplash(false), 2200);
    return () => clearTimeout(t);
  }, [superSport, bootSplash, loading, total]);

  // SUPER SPORT is a landscape-first, lean-back experience — request landscape where supported.
  useEffect(() => {
    if (!superSport) return;
    const orientation = (screen as unknown as { orientation?: { lock?: (o: string) => Promise<void> } }).orientation;
    orientation?.lock?.("landscape").catch(() => {
      /* unsupported / blocked outside fullscreen — CSS rotate hint covers portrait */
    });
  }, [superSport]);

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
      setHeaderRadioAudio(false);
    }
  }, [osdVisible]);

  useEffect(() => {
    if (!osdVisible) setChannelMenuOpen(false);
  }, [osdVisible]);

  useEffect(() => {
    if (!channelMenuOpen && !gearMenuOpen) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null;
      if (t?.closest(".lm-cable-channel-menu")) return;
      if (t?.closest(".lm-cable-gear-wrap")) return;
      if (t?.closest(".lm-cable-gear-menu")) return;
      if (t?.closest(".lm-cable-volume")) return;
      setChannelMenuOpen(false);
      setGearMenuOpen(false);
      setVolumeOpen(false);
      if (osdTimer.current) clearTimeout(osdTimer.current);
      if (!confirmRemoveOpen && !showRadioPage && !volumeOpen) {
        osdTimer.current = setTimeout(() => setOsdVisible(false), OSD_HIDE_MS);
      }
    };
    document.addEventListener("click", onDoc);
    return () => document.removeEventListener("click", onDoc);
  }, [channelMenuOpen, confirmRemoveOpen, gearMenuOpen, showRadioPage, volumeOpen]);

  useEffect(() => {
    if (!showRadioPage) return;
    setOsdVisible(true);
    if (osdTimer.current) clearTimeout(osdTimer.current);
  }, [showRadioPage]);

  useEffect(() => {
    if (showQuad || showRadioPage) setEpgOpen(false);
  }, [showQuad, showRadioPage, pageIndex]);

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
      if (welcomeOpen || epgOpen) return;
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
      if (gearMenuOpen && e.key === "Escape") {
        e.preventDefault();
        setGearMenuOpen(false);
        return;
      }
      if (volumeOpen && e.key === "Escape") {
        e.preventDefault();
        setVolumeOpen(false);
        return;
      }
      pokeOsd();
      if (showQuad && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        e.preventDefault();
        const d = rtl
          ? e.key === "ArrowLeft"
            ? 1
            : -1
          : e.key === "ArrowRight"
            ? 1
            : -1;
        setSelectedQuadSlot((s) => (s + d + 4) % 4);
        setAudioUnlocked(true);
        setHeaderRadioAudio(false);
        return;
      }
      if (e.key === "ArrowUp" || e.key === "PageUp") {
        e.preventDefault();
        changePage(1);
      }
      if (e.key === "ArrowDown" || e.key === "PageDown") {
        e.preventDefault();
        changePage(-1);
      }
      if (e.key === "Enter" && showQuad && focusHit) {
        e.preventDefault();
        jumpToFavoriteFromQuad(quadSlots[selectedQuadSlot] ?? 0);
        return;
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
  }, [
    changePage,
    channelMenuOpen,
    closeConfirmRemove,
    confirmRemoveOpen,
    epgOpen,
    focusHit,
    gearMenuOpen,
    goToQuad,
    jumpToFavoriteFromQuad,
    pokeOsd,
    quadSlots,
    rtl,
    selectedQuadSlot,
    showQuad,
    toggleFullscreen,
    volumeOpen,
    welcomeOpen,
  ]);

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
    return (
      <div className="lm-cable lm-cable--fullscreen">
        <div className="lm-cable-loading" aria-hidden="true">
          …
        </div>
        {welcomeOpen ? (
          <CableTunerWelcome
            uiLang={uiLang}
            booting
            channelCount={0}
            onStart={() => {
              dismissTvWelcome();
              setWelcomeOpen(false);
            }}
          />
        ) : null}
      </div>
    );
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
      className={`lm-cable lm-cable--fullscreen${isFullscreen ? " is-browser-fullscreen" : ""}${welcomeOpen ? " lm-cable--welcome-open" : ""}${superSport ? " lm-cable--supersport" : ""}${showQuad ? " lm-cable--quad" : " lm-cable--single-view"}`}
      dir={rtl ? "rtl" : "ltr"}
      onMouseMove={onPointerActivity}
      onTouchStart={onPointerActivity}
    >
      {superSport && bootSplash ? (
        <div className="lm-ss-splash" role="dialog" aria-modal="true" aria-label="SUPER SPORT">
          <div className="lm-ss-splash__glow" aria-hidden="true" />
          <div className="lm-ss-splash__brand">
            <span className="lm-ss-splash__super">SUPER</span>
            <span className="lm-ss-splash__sport">SPORT</span>
          </div>
          <p className="lm-ss-splash__tag">{rtl ? "חבילת הזהב · שידורי ספורט" : "GOLD PACKAGE · LIVE SPORTS"}</p>
          <div className="lm-ss-splash__bar" aria-hidden="true">
            <span className="lm-ss-splash__bar-fill" />
          </div>
          <p className="lm-ss-splash__status">
            {loading || total < 1
              ? rtl
                ? "מתחבר לשידור…"
                : "Connecting to broadcast…"
              : rtl
                ? `${total} ערוצי ספורט מוכנים`
                : `${total} sports channels ready`}
          </p>
        </div>
      ) : null}
      {superSport ? (
        <div className="lm-ss-rotate-hint" aria-hidden="true">
          <span className="lm-ss-rotate-hint__icon">📱↻</span>
          <span>{rtl ? "סובבו את המכשיר לרוחב לחוויית הצפייה המלאה" : "Rotate your device to landscape for the full experience"}</span>
        </div>
      ) : null}
      {welcomeOpen ? (
        <CableTunerWelcome
          uiLang={uiLang}
          booting={false}
          channelCount={total}
          onStart={() => {
            dismissTvWelcome();
            setWelcomeOpen(false);
            pokeOsd();
          }}
        />
      ) : null}
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
                rtl={rtl}
                selected={audioFocus}
                audioFocus={audioFocus && audioUnlocked && !userMuted && !headerRadioAudio}
                muted={slotMuted(audioFocus && !headerRadioAudio)}
                volume={volume}
                multiView
                loadTimeoutMs={CABLE_STREAM_LOAD_MS}
                quadJumpOpen={selectedQuadSlot === i && osdVisible && !globalSnow}
                quadJumpLabel={chNum > 0 ? L.jumpToChannel(chNum) : ""}
                onQuadJump={() => jumpToFavoriteFromQuad(quadSlots[i] ?? 0)}
                onDoubleActivate={() => jumpToFavoriteFromQuad(quadSlots[i] ?? 0)}
                onStreamReady={() => {
                  if (hit) markQuadStreamReady(hit.id);
                }}
                onStreamFail={() => handleQuadSlotFail(i)}
                onSelect={() => {
                  setSelectedQuadSlot(i);
                  setAudioUnlocked(true);
                  setHeaderRadioAudio(false);
                  pokeOsd();
                  // SUPER SPORT: a single tap opens the channel full-screen (with its EPG panel).
                  if (superSport) jumpToFavoriteFromQuad(quadSlots[i] ?? 0);
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
            rtl={rtl}
            audioFocus={audioUnlocked && !userMuted}
            muted={slotMuted(true)}
            volume={volume}
            single
            mediaRef={singleVideoRef}
            loadTimeoutMs={CABLE_STREAM_LOAD_MS}
            assumeReady={assumeSingleReady}
            onStreamFail={skipSingleToNext}
          />
          <LiveCaptionsOverlay
            original={captionsOriginal}
            translated={captionsTranslated}
            visible={captionsActive}
          />
        </div>
      ) : null}

      {superSport && showQuad && !bootSplash ? (
        <div className="lm-ss-quad-ui" dir="ltr">
          <button
            type="button"
            className="lm-ss-rail lm-ss-rail--left"
            onClick={() => changePage(-1)}
            disabled={globalSnow || total <= 1}
            aria-label={rtl ? "ערוצים קודמים" : "Previous channels"}
          >
            <span aria-hidden="true">‹</span>
          </button>
          <button
            type="button"
            className="lm-ss-rail lm-ss-rail--right"
            onClick={() => changePage(1)}
            disabled={globalSnow || total <= 1}
            aria-label={rtl ? "ערוצים הבאים" : "Next channels"}
          >
            <span aria-hidden="true">›</span>
          </button>

          <div className="lm-ss-quad-dock">
            <div className={`lm-ss-vol${volumeOpen ? " is-open" : ""}`}>
              <button
                type="button"
                className="lm-ss-dock-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setVolumeOpen((o) => !o);
                  setAudioUnlocked(true);
                  pokeOsd();
                }}
                aria-label={rtl ? "עוצמת קול" : "Volume"}
                aria-expanded={volumeOpen}
              >
                <span aria-hidden="true">{userMuted || volume === 0 ? "🔇" : volume < 0.45 ? "🔉" : "🔊"}</span>
              </button>
              <input
                type="range"
                className="lm-ss-vol-slider"
                min={0}
                max={100}
                value={Math.round((userMuted ? 0 : volume) * 100)}
                onChange={(e) => {
                  setAudioUnlocked(true);
                  setUserMuted(false);
                  onVolumeChange(Number(e.target.value) / 100);
                  pokeOsd();
                }}
                aria-label={rtl ? "עוצמת קול" : "Volume"}
              />
            </div>
            <button
              type="button"
              className="lm-ss-fs-pulse"
              onClick={() => void toggleFullscreen()}
              aria-label={isFullscreen ? (rtl ? "צא ממסך מלא" : "Exit full screen") : rtl ? "מסך מלא" : "Full screen"}
            >
              <span className="lm-ss-fs-pulse__icon" aria-hidden="true">
                {isFullscreen ? "⤢" : "⛶"}
              </span>
              <span className="lm-ss-fs-pulse__label">{isFullscreen ? (rtl ? "צא" : "Exit") : rtl ? "מסך מלא" : "Full screen"}</span>
            </button>
          </div>
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

      {epgOpen && epgHit ? (
        <CableEpgPanel
          favorites={favorites}
          categoryOrder={categoryOrder}
          focusHit={epgHit}
          uiLang={uiLang}
          onClose={() => setEpgOpen(false)}
        />
      ) : null}

      <div
        className={`lm-cable-osd${showRadioPage ? " lm-cable-osd--pinned" : osdVisible ? "" : " is-hidden"}${showQuad ? " lm-cable-osd--quad" : ""}${gearMenuOpen || volumeOpen ? " lm-cable-osd--popups-open" : ""}${rtl ? " lm-cable-osd--he" : ""}`}
        dir="ltr"
      >
        <div className="lm-cable-osd-toolbar">
          <button
            type="button"
            className="lm-cable-osd-tool"
            onClick={() => changePage(1)}
            disabled={globalSnow || (showRadioPage ? radioTotal <= 1 : total <= 1 && radioTotal === 0)}
            aria-label={L.chUpLbl}
          >
            <span className="lm-cable-osd-tool-icon" aria-hidden="true">
              {L.chUp}
            </span>
            <span className="lm-cable-osd-tool-label">{L.chUpLbl}</span>
          </button>
          <button
            type="button"
            className="lm-cable-osd-tool"
            onClick={() => changePage(-1)}
            disabled={globalSnow || (showRadioPage ? radioTotal <= 1 : total <= 1 && radioTotal === 0)}
            aria-label={L.chDownLbl}
          >
            <span className="lm-cable-osd-tool-icon" aria-hidden="true">
              {L.chDown}
            </span>
            <span className="lm-cable-osd-tool-label">{L.chDownLbl}</span>
          </button>
          {!showQuad ? (
            <button
              type="button"
              className="lm-cable-osd-tool"
              onClick={goToQuad}
              disabled={globalSnow}
              aria-label={L.menuLbl}
              title={L.backQuad}
            >
              <span className="lm-cable-osd-tool-icon" aria-hidden="true">
                ⊞
              </span>
              <span className="lm-cable-osd-tool-label">{L.menuLbl}</span>
            </button>
          ) : null}
          {epgHit && favorites.length > 0 ? (
            <button
              type="button"
              className="lm-cable-osd-tool lm-cable-osd-tool--epg"
              onClick={() => {
                setEpgOpen(true);
                pokeOsd();
              }}
              disabled={globalSnow}
              aria-label={L.epg}
              title={
                epgGuide.loading
                  ? L.epg
                  : `${L.epg}${epgGuide.readyCount ? ` (${epgGuide.readyCount})` : ""}`
              }
            >
              <span className="lm-cable-osd-tool-icon" aria-hidden="true">
                ▦
              </span>
              <span className="lm-cable-osd-tool-label">
                EPG{epgGuide.readyCount > 0 ? ` ${epgGuide.readyCount}` : ""}
              </span>
            </button>
          ) : null}
          <button type="button" className="lm-cable-osd-tool" onClick={onOpenBrowse}>
            <span className="lm-cable-osd-tool-icon" aria-hidden="true">
              ⌕
            </span>
            <span className="lm-cable-osd-tool-label">{L.browse}</span>
          </button>
          <button
            type="button"
            className="lm-cable-osd-tool"
            onClick={() => void toggleFullscreen()}
            aria-label={isFullscreen ? L.exitFullscreen : L.fullscreenLbl}
            title={isFullscreen ? L.exitFullscreen : L.fullscreenLbl}
          >
            <span className="lm-cable-osd-tool-icon" aria-hidden="true">
              {isFullscreen ? "⤢" : "⛶"}
            </span>
            <span className="lm-cable-osd-tool-label">{L.fullscreenLbl}</span>
          </button>
          {singleChannelView ? (
            <div className="lm-cable-gear-wrap">
              <button
                ref={gearBtnRef}
                type="button"
                className={`lm-cable-osd-tool${gearMenuOpen ? " is-active" : ""}${captionsActive ? " lm-cable-osd-tool--captions-on" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setGearMenuOpen((o) => !o);
                  setVolumeOpen(false);
                  pokeOsd();
                }}
                aria-label={L.settingsLbl}
                aria-expanded={gearMenuOpen}
                title={L.settingsLbl}
              >
                <span className="lm-cable-osd-tool-icon" aria-hidden="true">
                  ⚙
                </span>
                <span className="lm-cable-osd-tool-label">{L.settingsLbl}</span>
              </button>
            </div>
          ) : null}
          {singleChannelView ? (
            <CableTunerGearMenu
              uiLang={uiLang}
              anchorRef={gearBtnRef}
              open={gearMenuOpen}
              captionsActive={captionsActive}
              captionsStatus={captionsStatus}
              statusMessage={captionsStatusMessage}
              loadPct={captionsLoadPct}
              sourceLang={captionSourceLang}
              targetLang={captionTargetLang}
              captionsSupported={captionsSupported}
              onSourceLang={setCaptionSourceLang}
              onTargetLang={setCaptionTargetLang}
              onToggleCaptions={toggleCaptions}
              onClose={() => setGearMenuOpen(false)}
            />
          ) : null}
          <div className={`lm-cable-volume${volumeOpen ? " is-open" : ""}`} title={L.volumeLbl}>
            <button
              type="button"
              className="lm-cable-osd-tool lm-cable-volume-trigger"
              onClick={(e) => {
                e.stopPropagation();
                setVolumeOpen((o) => !o);
                setGearMenuOpen(false);
                setAudioUnlocked(true);
                pokeOsd();
              }}
              aria-label={L.volumeLbl}
              aria-expanded={volumeOpen}
            >
              <span className="lm-cable-osd-tool-icon" aria-hidden="true">
                {userMuted || volume === 0 ? "🔇" : volume < 0.45 ? "🔉" : "🔊"}
              </span>
              <span className="lm-cable-osd-tool-label">{L.volumeLbl}</span>
            </button>
            <div className="lm-cable-volume-rail">
              <input
                type="range"
                className="lm-cable-volume-slider"
                min={0}
                max={100}
                value={Math.round((userMuted ? 0 : volume) * 100)}
                onChange={(e) => {
                  setAudioUnlocked(true);
                  setUserMuted(false);
                  onVolumeChange(Number(e.target.value) / 100);
                  pokeOsd();
                }}
                onPointerDown={() => pokeOsd()}
                aria-label={L.volume}
              />
              <button
                type="button"
                className="lm-cable-volume-mute-btn"
                onClick={() => {
                  setAudioUnlocked(true);
                  setUserMuted((m) => !m);
                  pokeOsd();
                }}
                aria-label={userMuted ? L.unmute : L.mute}
                title={userMuted ? L.unmute : L.mute}
              >
                {userMuted ? "🔇" : "🔊"}
              </button>
            </div>
          </div>
        </div>

        <div className={`lm-cable-osd-panel${osdNoProgram ? " lm-cable-osd-panel--no-epg" : ""}`}>
          <div className="lm-cable-osd-col lm-cable-osd-col--channel" dir={rtl ? "rtl" : "ltr"}>
            {osdChannelNum != null ? (
              <span className="lm-cable-osd-ch-num">{osdChannelNum}</span>
            ) : osdChannelRange ? (
              <span className="lm-cable-osd-ch-num lm-cable-osd-ch-num--range">{osdChannelRange}</span>
            ) : (
              <span className="lm-cable-osd-ch-num">—</span>
            )}
            {!osdNoProgram ? (
              <span className="lm-cable-osd-ch-name" title={focusHit?.title}>
                {showQuad && range ? L.range(range.from, range.to) : showQuad ? L.quadScreen : centerTitle || "—"}
              </span>
            ) : null}
          </div>

          <div className="lm-cable-osd-col lm-cable-osd-col--program" dir={rtl ? "rtl" : "ltr"}>
            {showQuad ? (
              <div className="lm-cable-osd-quad-summary">
                <div className="lm-cable-osd-channel lm-cable-osd-channel--quad" role="list">
                  {quadTiles.map((hit, i) => {
                    const chNum = quadSlots[i] != null ? quadSlots[i] + 1 : 0;
                    const title = hit ? shortenCableChannelTitle(hit.title, 16) : "—";
                    const active = selectedQuadSlot === i;
                    return (
                      <button
                        key={`quad-ch-${i}`}
                        type="button"
                        role="listitem"
                        className={`lm-cable-osd-channel-part${active ? " is-active" : ""}`}
                        title={hit?.title}
                        onClick={() => {
                          setSelectedQuadSlot(i);
                          setAudioUnlocked(true);
                          setHeaderRadioAudio(false);
                          pokeOsd();
                        }}
                        onDoubleClick={(e) => {
                          e.preventDefault();
                          jumpToFavoriteFromQuad(quadSlots[i] ?? 0);
                        }}
                      >
                        {chNum > 0 ? `${String(chNum).padStart(2, "0")} ${title}` : title}
                      </button>
                    );
                  })}
                </div>
                <p className="lm-cable-osd-quad-audio-hint">
                  {focusHit ? L.quadAudioOn(centerTitle) : L.quadHint}
                </p>
              </div>
            ) : nowPlaying ? (
              <div className="lm-cable-osd-program" aria-live="polite">
                {nowPlaying.posterUrl ? (
                  <img
                    className="lm-cable-osd-now-poster"
                    src={nowPlaying.posterUrl}
                    alt=""
                    loading="lazy"
                  />
                ) : null}
                <div className="lm-cable-osd-program-body">
                  <span className="lm-cable-osd-now-kicker">
                    {nowPlaying.seriesTitle && nowPlaying.seriesTitle !== nowPlaying.displayTitle ?
                      nowPlaying.seriesTitle
                    : L.onNow}
                  </span>
                  <p className="lm-cable-osd-now-title" title={nowPlaying.displayTitle}>
                    {nowPlaying.displayTitle}
                  </p>
                  {nowPlaying.episodeLabel ? (
                    <p className="lm-cable-osd-now-episode">{nowPlaying.episodeLabel}</p>
                  ) : null}
                  {nowPlaying.description ? (
                    <p className="lm-cable-osd-now-desc" title={nowPlaying.description}>
                      {nowPlaying.description}
                    </p>
                  ) : null}
                </div>
              </div>
            ) : (
              <div className="lm-cable-osd-program lm-cable-osd-program--idle">
                {osdNoProgram ? (
                  <p className="lm-cable-osd-idle-title lm-cable-osd-idle-title--centered" title={focusHit?.title}>
                    {centerTitle || "—"}
                  </p>
                ) : (
                  <p className="lm-cable-osd-idle-title">{centerTitle || L.tuning}</p>
                )}
                {globalSnow ? <p className="lm-cable-osd-tuning">{L.tuning}</p> : null}
              </div>
            )}
          </div>

          <div className="lm-cable-osd-col lm-cable-osd-col--timing">
            {showQuad ? (
              <button
                type="button"
                className="lm-cable-osd-btn lm-cable-osd-btn--danger lm-cable-osd-quad-jump"
                disabled={!focusHit}
                onClick={() => jumpToFavoriteFromQuad(quadSlots[selectedQuadSlot] ?? 0)}
              >
                {focusHit ? L.jumpToChannel((quadSlots[selectedQuadSlot] ?? 0) + 1) : L.quadPickChannel}
              </button>
            ) : nowPlaying ? (
              <>
                <span className="lm-cable-osd-time-range">
                  {formatOsdProgramRange(nowPlaying.displayStart, nowPlaying.displayEnd, rtl)}
                </span>
                <span className="lm-cable-osd-now-ends">{L.endsIn(nowPlaying.minutesLeft)}</span>
                <div className="lm-cable-osd-timing-meta">
                  <span className="lm-cable-osd-now-duration">
                    {formatDurationMinutes(nowPlaying.durationMinutes, rtl)}
                  </span>
                  {nowPlaying.tmdbYear ? (
                    <span className="lm-cable-osd-now-year">{nowPlaying.tmdbYear}</span>
                  ) : null}
                  {nowPlaying.tmdbRating != null ? (
                    <span className="lm-cable-osd-now-rating">★ {nowPlaying.tmdbRating.toFixed(1)}</span>
                  ) : null}
                </div>
                <div className="lm-cable-osd-now-bar" aria-hidden="true">
                  <span className="lm-cable-osd-now-fill" style={{ width: `${nowPlaying.progressPct}%` }} />
                </div>
              </>
            ) : null}
          </div>

          <div className="lm-cable-osd-col lm-cable-osd-col--status" dir={rtl ? "rtl" : "ltr"}>
            <span className="lm-cable-osd-clock">{clock}</span>
            <span className="lm-cable-osd-date">{dateLabel}</span>
            {weatherLabel ? <span className="lm-cable-osd-weather">{weatherLabel}</span> : null}
          </div>
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
