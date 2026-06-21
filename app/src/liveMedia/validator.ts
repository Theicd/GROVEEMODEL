import type { StreamStatus } from "./types";

export async function validateStream(url: string, timeoutMs = 8000): Promise<StreamStatus> {
  return (await validateStreamWithMetrics(url, timeoutMs)).status;
}

export async function validateRadioStream(url: string, timeoutMs = 8000): Promise<StreamStatus> {
  return (await validateRadioStreamWithMetrics(url, timeoutMs)).status;
}

export async function validateStreamWithMetrics(
  url: string,
  timeoutMs = 8000,
): Promise<{ status: StreamStatus; loadTimeMs: number }> {
  const start = Date.now();
  try {
    if (/\.m3u8(\?|$)/i.test(url) || url.includes("m3u8")) {
      const status = await probeHLS(url, timeoutMs);
      return { status, loadTimeMs: Date.now() - start };
    }
    const status = await probeMediaElement(url, timeoutMs, "video");
    return { status, loadTimeMs: Date.now() - start };
  } catch {
    return { status: "offline", loadTimeMs: Date.now() - start };
  }
}

export async function validateRadioStreamWithMetrics(
  url: string,
  timeoutMs = 8000,
): Promise<{ status: StreamStatus; loadTimeMs: number }> {
  const start = Date.now();
  try {
    const status = await probeMediaElement(url, timeoutMs, "audio");
    return { status, loadTimeMs: Date.now() - start };
  } catch {
    return { status: "offline", loadTimeMs: Date.now() - start };
  }
}

async function probeHLS(url: string, timeoutMs: number): Promise<StreamStatus> {
  if (typeof window === "undefined") return "unknown";
  try {
    const Hls = (await import("hls.js")).default;
    if (!Hls.isSupported()) return probeMediaElement(url, timeoutMs, "video");
    return new Promise<StreamStatus>((resolve) => {
      let resolved = false;
      const finish = (s: StreamStatus) => {
        if (resolved) return;
        resolved = true;
        try {
          hls.destroy();
        } catch {
          /* ignore */
        }
        resolve(s);
      };
      const timer = setTimeout(() => finish("warning"), timeoutMs);
      const video = document.createElement("video");
      const hls = new Hls({ enableWorker: false, lowLatencyMode: true });
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        clearTimeout(timer);
        finish("working");
      });
      hls.on(Hls.Events.ERROR, (_e, data) => {
        clearTimeout(timer);
        if (data.fatal) finish("offline");
      });
      hls.loadSource(url);
    });
  } catch {
    return "offline";
  }
}

async function probeMediaElement(
  url: string,
  timeoutMs: number,
  tag: "video" | "audio",
): Promise<StreamStatus> {
  if (typeof window === "undefined") return "unknown";
  return new Promise<StreamStatus>((resolve) => {
    const el = document.createElement(tag);
    el.preload = "none";
    el.muted = true;
    let done = false;
    const finish = (s: StreamStatus) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      el.removeAttribute("src");
      el.load();
      resolve(s);
    };
    const timer = setTimeout(() => finish("warning"), timeoutMs);
    el.addEventListener("loadedmetadata", () => finish("working"));
    el.addEventListener("canplay", () => finish("working"));
    el.addEventListener("error", () => finish("offline"));
    el.src = url;
    el.load();
  });
}
