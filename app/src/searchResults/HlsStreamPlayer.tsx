import { useEffect, useRef } from "react";

type Props = {
  src: string;
  poster?: string;
  className?: string;
  autoPlay?: boolean;
  tag?: "video" | "audio";
  controls?: boolean;
  muted?: boolean;
  volume?: number;
  /** Lighter HLS settings when several players run together (quad view). */
  multiView?: boolean;
  onCanPlay?: () => void;
  onStreamFail?: () => void;
};

export function HlsStreamPlayer({
  src,
  poster,
  className,
  autoPlay = true,
  tag = "video",
  controls = true,
  muted = false,
  volume = 1,
  multiView = false,
  onCanPlay,
  onStreamFail,
}: Props) {
  const ref = useRef<HTMLVideoElement & HTMLAudioElement>(null);
  const onCanPlayRef = useRef(onCanPlay);
  onCanPlayRef.current = onCanPlay;
  const onStreamFailRef = useRef(onStreamFail);
  onStreamFailRef.current = onStreamFail;

  useEffect(() => {
    const el = ref.current;
    if (!el || !src) return;

    let failed = false;
    const failOnce = () => {
      if (failed) return;
      failed = true;
      onStreamFailRef.current?.();
    };

    const notify = () => onCanPlayRef.current?.();
    const notifyIfReady = () => {
      if (el.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) notify();
    };

    el.addEventListener("canplay", notify);
    el.addEventListener("playing", notify);
    el.addEventListener("loadeddata", notifyIfReady);
    el.addEventListener("error", failOnce);

    let hls: import("hls.js").default | null = null;
    let cancelled = false;

    const isHls = /\.m3u8(\?|$)/i.test(src) || src.includes("m3u8");

    const tryPlay = () => {
      if (!cancelled && autoPlay) void (el as HTMLVideoElement).play().catch(() => {});
    };

    void (async () => {
      if (isHls && tag === "video") {
        try {
          const Hls = (await import("hls.js")).default;
          if (Hls.isSupported()) {
            hls = new Hls({
              enableWorker: true,
              lowLatencyMode: !multiView,
              maxBufferLength: multiView ? 10 : 30,
              maxMaxBufferLength: multiView ? 15 : 600,
              capLevelToPlayerSize: multiView,
            });
            hls.loadSource(src);
            hls.attachMedia(el as HTMLVideoElement);
            hls.on(Hls.Events.MANIFEST_PARSED, () => {
              tryPlay();
              notifyIfReady();
            });
            let fatalRetries = 0;
            hls.on(Hls.Events.ERROR, (_, data) => {
              if (!data.fatal) return;
              if (fatalRetries < 1) {
                fatalRetries += 1;
                try {
                  hls?.startLoad();
                } catch {
                  failOnce();
                }
                return;
              }
              failOnce();
            });
            notifyIfReady();
            return;
          }
        } catch {
          /* fall through */
        }
      }
      el.src = src;
      tryPlay();
      notifyIfReady();
    })();

    const retryPlay = window.setInterval(() => {
      if (cancelled || el.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) return;
      tryPlay();
    }, multiView ? 1500 : 3000);

    return () => {
      cancelled = true;
      window.clearInterval(retryPlay);
      el.removeEventListener("canplay", notify);
      el.removeEventListener("playing", notify);
      el.removeEventListener("loadeddata", notifyIfReady);
      el.removeEventListener("error", failOnce);
      if (hls) {
        try {
          hls.destroy();
        } catch {
          /* ignore */
        }
      }
      el.removeAttribute("src");
      el.load();
    };
  }, [src, autoPlay, tag, multiView]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.volume = Math.max(0, Math.min(1, volume));
  }, [volume]);

  useEffect(() => {
    const el = ref.current;
    if (!el || muted) return;
    void el.play().catch(() => {});
  }, [muted]);

  if (tag === "audio") {
    return (
      <audio
        ref={ref as React.RefObject<HTMLAudioElement>}
        className={className}
        controls={controls}
        autoPlay={autoPlay}
        muted={muted}
        playsInline
      />
    );
  }

  return (
    <video
      ref={ref as React.RefObject<HTMLVideoElement>}
      className={className}
      controls={controls}
      autoPlay={autoPlay}
      muted={muted}
      playsInline
      poster={poster}
      disablePictureInPicture
      controlsList={controls ? undefined : "nofullscreen nodownload noremoteplayback"}
      onDoubleClick={(e) => {
        if (!controls) e.preventDefault();
      }}
    />
  );
}
