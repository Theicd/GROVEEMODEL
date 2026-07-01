/**
 * Live-video element pool for the cable tuner.
 *
 * The quad grid and the single-channel view are different parts of the React tree,
 * so navigating between them would normally unmount one <video> and mount another —
 * forcing a full HLS reload (a black screen for a few seconds) even though the channel
 * was already playing.
 *
 * This pool keeps the actual <video> element (with its attached, *still playing* hls.js
 * instance) alive and simply MOVES the DOM node between the on-screen slot and a hidden
 * parking area. Relocating a playing <video> in the DOM does not interrupt playback, so
 * clicking a quad tile opens the channel instantly at the live edge.
 *
 * Entries are keyed by stream src. Parked entries keep playing (muted) for a short TTL
 * so quick back-and-forth stays instant, then are destroyed to free bandwidth.
 */

type HlsInstance = import("hls.js").default;

export type PooledVideo = {
  src: string;
  video: HTMLVideoElement;
  hls: HlsInstance | null;
  inUse: boolean;
  multiView: boolean;
  parkTimer: number | null;
  ready: boolean;
  onReady: (() => void) | null;
  onFail: (() => void) | null;
};

const PARK_TTL_MS = 30_000;
const MAX_PARKED = 4;

const pool = new Map<string, PooledVideo>();

let HlsCtor: typeof import("hls.js").default | null = null;
let hlsLoad: Promise<void> | null = null;
function ensureHls(): Promise<void> {
  if (HlsCtor) return Promise.resolve();
  if (!hlsLoad) {
    hlsLoad = import("hls.js")
      .then((m) => {
        HlsCtor = m.default;
      })
      .catch(() => {
        HlsCtor = null;
      });
  }
  return hlsLoad;
}

let lot: HTMLDivElement | null = null;
function parkingLot(): HTMLDivElement {
  if (lot && lot.isConnected) return lot;
  lot = document.createElement("div");
  lot.setAttribute("data-cable-video-parking", "");
  lot.style.cssText =
    "position:fixed;left:-99999px;top:-99999px;width:1px;height:1px;overflow:hidden;opacity:0;pointer-events:none;";
  document.body.appendChild(lot);
  return lot;
}

function isHlsSrc(src: string): boolean {
  return /\.m3u8(\?|$)/i.test(src) || src.includes("m3u8");
}

function createVideo(src: string, multiView: boolean): PooledVideo {
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.setAttribute("playsinline", "");
  video.setAttribute("disablepictureinpicture", "");
  video.controls = false;

  const entry: PooledVideo = {
    src,
    video,
    hls: null,
    inUse: true,
    multiView,
    parkTimer: null,
    ready: false,
    onReady: null,
    onFail: null,
  };

  const markReady = () => {
    entry.ready = true;
    entry.onReady?.();
  };
  const failOnce = (() => {
    let failed = false;
    return () => {
      if (failed) return;
      failed = true;
      entry.onFail?.();
    };
  })();

  video.addEventListener("canplay", markReady);
  video.addEventListener("playing", markReady);
  video.addEventListener("loadeddata", () => {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) markReady();
  });
  video.addEventListener("error", failOnce);

  const tryPlay = () => {
    void video.play().catch(() => {});
  };

  void (async () => {
    if (isHlsSrc(src)) {
      await ensureHls();
      if (HlsCtor && HlsCtor.isSupported()) {
        const hls = new HlsCtor({
          enableWorker: true,
          lowLatencyMode: !multiView,
          maxBufferLength: multiView ? 10 : 30,
          maxMaxBufferLength: multiView ? 15 : 600,
          capLevelToPlayerSize: multiView,
        });
        entry.hls = hls;
        hls.loadSource(src);
        hls.attachMedia(video);
        hls.on(HlsCtor.Events.MANIFEST_PARSED, () => {
          tryPlay();
          if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) markReady();
        });
        let fatalRetries = 0;
        hls.on(HlsCtor.Events.ERROR, (_evt, data) => {
          if (!data?.fatal) return;
          if (fatalRetries < 1) {
            fatalRetries += 1;
            try {
              hls.startLoad();
            } catch {
              failOnce();
            }
            return;
          }
          failOnce();
        });
        return;
      }
    }
    video.src = src;
    tryPlay();
  })();

  return entry;
}

function destroy(entry: PooledVideo) {
  if (entry.parkTimer) {
    clearTimeout(entry.parkTimer);
    entry.parkTimer = null;
  }
  try {
    entry.hls?.destroy();
  } catch {
    /* ignore */
  }
  try {
    entry.video.pause();
    entry.video.removeAttribute("src");
    entry.video.load();
    entry.video.remove();
  } catch {
    /* ignore */
  }
  if (pool.get(entry.src) === entry) pool.delete(entry.src);
}

function enforceParkLimit() {
  const parked = [...pool.values()].filter((e) => !e.inUse);
  if (parked.length <= MAX_PARKED) return;
  // Oldest park timers destroyed first is hard to know; just trim extras.
  const extra = parked.slice(0, parked.length - MAX_PARKED);
  for (const e of extra) destroy(e);
}

/**
 * Get a live <video> for `src`, reusing a parked (still-playing) element when available.
 * The caller is responsible for appending `entry.video` into its container.
 */
export function acquireCableVideo(
  src: string,
  multiView: boolean,
  handlers: { onReady?: () => void; onFail?: () => void },
): PooledVideo {
  const existing = pool.get(src);
  if (existing) {
    if (existing.parkTimer) {
      clearTimeout(existing.parkTimer);
      existing.parkTimer = null;
    }
    existing.inUse = true;
    existing.multiView = multiView;
    existing.onReady = handlers.onReady ?? null;
    existing.onFail = handlers.onFail ?? null;
    // Reused element is typically already live — surface readiness on next tick.
    if (existing.ready || existing.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      queueMicrotask(() => existing.onReady?.());
    }
    void existing.video.play().catch(() => {});
    return existing;
  }
  const entry = createVideo(src, multiView);
  entry.onReady = handlers.onReady ?? null;
  entry.onFail = handlers.onFail ?? null;
  pool.set(src, entry);
  return entry;
}

/** Park the element back off-screen (keeps playing, muted) with a destroy TTL. */
export function releaseCableVideo(entry: PooledVideo): void {
  entry.inUse = false;
  entry.onReady = null;
  entry.onFail = null;
  try {
    entry.video.muted = true;
    parkingLot().appendChild(entry.video);
  } catch {
    /* ignore */
  }
  if (entry.parkTimer) clearTimeout(entry.parkTimer);
  entry.parkTimer = window.setTimeout(() => {
    if (!entry.inUse) destroy(entry);
  }, PARK_TTL_MS);
  enforceParkLimit();
}
