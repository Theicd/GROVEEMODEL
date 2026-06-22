import { useEffect, useRef } from "react";

type Props = {
  src: string;
  poster?: string;
  className?: string;
  autoPlay?: boolean;
  tag?: "video" | "audio";
};

export function HlsStreamPlayer({ src, poster, className, autoPlay = true, tag = "video" }: Props) {
  const ref = useRef<HTMLVideoElement & HTMLAudioElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !src) return;

    let hls: import("hls.js").default | null = null;
    let cancelled = false;

    const isHls = /\.m3u8(\?|$)/i.test(src) || src.includes("m3u8");

    void (async () => {
      if (isHls && tag === "video") {
        try {
          const Hls = (await import("hls.js")).default;
          if (Hls.isSupported()) {
            hls = new Hls({ enableWorker: true, lowLatencyMode: true });
            hls.loadSource(src);
            hls.attachMedia(el as HTMLVideoElement);
            hls.on(Hls.Events.MANIFEST_PARSED, () => {
              if (!cancelled && autoPlay) void (el as HTMLVideoElement).play().catch(() => {});
            });
            return;
          }
        } catch {
          /* fall through */
        }
      }
      el.src = src;
      if (autoPlay) void el.play().catch(() => {});
    })();

    return () => {
      cancelled = true;
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
  }, [src, autoPlay, tag]);

  if (tag === "audio") {
    return (
      <audio
        ref={ref as React.RefObject<HTMLAudioElement>}
        className={className}
        controls
        autoPlay={autoPlay}
        playsInline
      />
    );
  }

  return (
    <video
      ref={ref as React.RefObject<HTMLVideoElement>}
      className={className}
      controls
      autoPlay={autoPlay}
      playsInline
      poster={poster}
    />
  );
}
