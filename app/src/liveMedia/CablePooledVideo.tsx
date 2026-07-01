import { useEffect, useRef, type RefObject } from "react";
import { acquireCableVideo, releaseCableVideo, type PooledVideo } from "./cableVideoPool";

type Props = {
  src: string;
  className?: string;
  muted?: boolean;
  volume?: number;
  multiView?: boolean;
  /** Expose the underlying <video> (single-channel captions, etc.). */
  mediaRef?: RefObject<HTMLVideoElement | null>;
  onCanPlay?: () => void;
  onStreamFail?: () => void;
};

/**
 * Renders a channel using the shared {@link acquireCableVideo} pool so the same
 * playing <video> element survives quad ↔ single transitions (instant switching).
 */
export function CablePooledVideo({
  src,
  className,
  muted = true,
  volume = 1,
  multiView = false,
  mediaRef,
  onCanPlay,
  onStreamFail,
}: Props) {
  const holderRef = useRef<HTMLDivElement>(null);
  const entryRef = useRef<PooledVideo | null>(null);
  const onCanPlayRef = useRef(onCanPlay);
  onCanPlayRef.current = onCanPlay;
  const onStreamFailRef = useRef(onStreamFail);
  onStreamFailRef.current = onStreamFail;

  useEffect(() => {
    const holder = holderRef.current;
    if (!holder || !src) return;

    const entry = acquireCableVideo(src, multiView, {
      onReady: () => onCanPlayRef.current?.(),
      onFail: () => onStreamFailRef.current?.(),
    });
    entryRef.current = entry;
    entry.video.className = className ?? "";
    holder.appendChild(entry.video);
    if (mediaRef) mediaRef.current = entry.video;

    return () => {
      if (mediaRef && mediaRef.current === entry.video) mediaRef.current = null;
      entryRef.current = null;
      releaseCableVideo(entry);
    };
    // Only src/multiView identity should re-acquire; other props are applied below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, multiView]);

  useEffect(() => {
    const entry = entryRef.current;
    if (!entry) return;
    entry.video.muted = muted;
    entry.video.volume = Math.max(0, Math.min(1, volume));
    if (!muted) void entry.video.play().catch(() => {});
  }, [muted, volume]);

  return <div ref={holderRef} className="lm-cable-pooled-holder" />;
}
