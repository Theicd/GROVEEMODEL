import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { UnifiedSearchHit } from "../searchResults/types";
import { HlsStreamPlayer } from "../searchResults/HlsStreamPlayer";
import { CablePooledVideo } from "./CablePooledVideo";
import { CABLE_STREAM_LOAD_MS } from "./cableTunerUtils";
import { TvStaticOverlay } from "./TvStaticOverlay";

type Props = {
  hit: UnifiedSearchHit | null;
  globalSnow: boolean;
  osdVisible: boolean;
  channelNum: number;
  preload?: boolean;
  single?: boolean;
  selected?: boolean;
  audioFocus?: boolean;
  muted?: boolean;
  volume?: number;
  multiView?: boolean;
  loadTimeoutMs?: number;
  channelBadgeTopRight?: boolean;
  /** Ref to video element (single-channel view only). */
  mediaRef?: RefObject<HTMLVideoElement | null>;
  quadJumpOpen?: boolean;
  quadJumpLabel?: string;
  onSelect?: () => void;
  onQuadJump?: () => void;
  onDoubleActivate?: () => void;
  /** Fired when the stream goes live; `elapsedMs` is the measured time-to-first-frame. */
  onStreamReady?: (elapsedMs: number) => void;
  onStreamFail?: () => void;
  /** Skip tuning snow when the same stream was already playing (e.g. quad → full screen). */
  assumeReady?: boolean;
};

export function CableStreamSlot({
  hit,
  globalSnow,
  osdVisible,
  channelNum,
  preload = false,
  single = false,
  selected = false,
  audioFocus = false,
  muted = true,
  volume = 1,
  multiView = false,
  loadTimeoutMs = CABLE_STREAM_LOAD_MS,
  channelBadgeTopRight = false,
  mediaRef,
  quadJumpOpen = false,
  quadJumpLabel = "",
  onSelect,
  onQuadJump,
  onDoubleActivate,
  onStreamReady,
  onStreamFail,
  assumeReady = false,
}: Props) {
  const [signalReady, setSignalReady] = useState(() => assumeReady);
  const failedRef = useRef(false);
  const startRef = useRef(0);
  const src = hit?.mediaPlayUrl || hit?.url || "";

  useEffect(() => {
    failedRef.current = false;
    startRef.current = (typeof performance !== "undefined" ? performance.now() : Date.now());
    if (!hit || !src) {
      setSignalReady(false);
      return;
    }
    if (assumeReady) {
      setSignalReady(true);
      return;
    }
    setSignalReady(false);
  }, [assumeReady, hit, src]);

  const onStreamReadyInternal = useCallback(() => {
    setSignalReady(true);
    const nowT = typeof performance !== "undefined" ? performance.now() : Date.now();
    const elapsed = startRef.current ? Math.max(0, nowT - startRef.current) : 0;
    onStreamReady?.(elapsed);
  }, [onStreamReady]);

  const onStreamFailInternal = useCallback(() => {
    if (failedRef.current) return;
    failedRef.current = true;
    onStreamFail?.();
  }, [onStreamFail]);

  useEffect(() => {
    if (!hit || signalReady || globalSnow || failedRef.current) return;
    const failTimer = window.setTimeout(() => {
      if (!signalReady) onStreamFailInternal();
    }, loadTimeoutMs);
    return () => window.clearTimeout(failTimer);
  }, [globalSnow, hit, loadTimeoutMs, onStreamFailInternal, signalReady, src]);

  if (!hit) {
    return (
      <div className="lm-cable-tile lm-cable-tile--empty">
        <div className="lm-cable-tile-screen" />
      </div>
    );
  }

  if (preload) {
    return (
      <div className="lm-cable-preload-slot" aria-hidden="true">
        <HlsStreamPlayer
          key={`${hit.id}-${src}`}
          src={src}
          muted
          controls={false}
          autoPlay
          multiView
          className="lm-cable-preload-video"
          onCanPlay={onStreamReadyInternal}
          onStreamFail={onStreamFailInternal}
        />
      </div>
    );
  }

  const showSnow = globalSnow || !signalReady;

  return (
    <div
      className={`lm-cable-tile${single ? " lm-cable-tile--single" : ""}${selected ? " is-selected" : ""}${audioFocus ? " is-audio-focus" : ""}${onSelect ? " is-selectable" : ""}${showSnow ? " is-tuning" : " is-locked"}`}
      onClick={onSelect}
      onDoubleClick={
        onDoubleActivate
          ? (e) => {
              e.preventDefault();
              onDoubleActivate();
            }
          : undefined
      }
      onKeyDown={
        onSelect
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect();
              }
            }
          : undefined
      }
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
    >
      <div className="lm-cable-tile-screen">
        <CablePooledVideo
          key={src}
          src={src}
          muted={muted}
          volume={volume}
          multiView={multiView}
          mediaRef={single ? mediaRef : undefined}
          className="lm-cable-tile-video"
          onCanPlay={onStreamReadyInternal}
          onStreamFail={onStreamFailInternal}
        />
        <TvStaticOverlay active={showSnow} />
        {audioFocus && !showSnow ? (
          <div className="lm-cable-tile-audio" aria-hidden="true">
            🔊
          </div>
        ) : null}
        {osdVisible && channelNum > 0 ? (
          <div className={`lm-cable-tile-osd${channelBadgeTopRight ? " lm-cable-tile-osd--tr" : ""}`}>
            <span className="lm-cable-tile-ch">{String(channelNum).padStart(2, "0")}</span>
          </div>
        ) : null}
        {quadJumpOpen && onQuadJump && !showSnow ? (
          <button
            type="button"
            className="lm-cable-tile-jump"
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              onQuadJump();
            }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            {quadJumpLabel}
          </button>
        ) : null}
      </div>
    </div>
  );
}
