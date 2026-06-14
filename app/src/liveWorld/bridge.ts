import { ingestGlobeLivePayload } from "./fetchSnapshot";
import type { LiveWorldSnapshot } from "./types";

type LiveSnapshotRequest = {
  resolve: (snap: LiveWorldSnapshot | null) => void;
  timer: ReturnType<typeof setTimeout>;
};

const pending = new Map<string, LiveSnapshotRequest>();
let requestSeq = 0;

export function registerGlobeLiveSnapshotListener(): () => void {
  const onMsg = (e: MessageEvent) => {
    if (e.data?.source !== "reality-core" || e.data?.type !== "live_snapshot") return;
    const snap = ingestGlobeLivePayload(e.data.payload);
    for (const [id, req] of pending) {
      clearTimeout(req.timer);
      req.resolve(snap);
      pending.delete(id);
    }
  };
  window.addEventListener("message", onMsg);
  return () => window.removeEventListener("message", onMsg);
}

/** Ask open Reality iframe for its live.* cache (3s timeout). */
export function requestLiveSnapshotFromGlobe(
  iframe: HTMLIFrameElement | null,
  timeoutMs = 3500,
): Promise<LiveWorldSnapshot | null> {
  if (!iframe?.contentWindow) return Promise.resolve(null);

  return new Promise((resolve) => {
    const id = `lw-${++requestSeq}`;
    const timer = setTimeout(() => {
      pending.delete(id);
      resolve(null);
    }, timeoutMs);

    pending.set(id, { resolve, timer });

    iframe.contentWindow!.postMessage(
      { source: "grovee", type: "getLiveSnapshot", requestId: id },
      "*",
    );
  });
}

export function sendGlobeGetLiveSnapshot(iframe: HTMLIFrameElement | null): void {
  if (!iframe?.contentWindow) return;
  iframe.contentWindow.postMessage({ source: "grovee", type: "getLiveSnapshot" }, "*");
}

export function findGlobeIframe(): HTMLIFrameElement | null {
  if (typeof document === "undefined") return null;
  return document.querySelector("iframe.globe-embed-frame") as HTMLIFrameElement | null;
}

/** Fire-and-forget: ask open Reality iframe to push live.* into snapshot cache. */
export function pingGlobeForLiveSnapshot(): void {
  sendGlobeGetLiveSnapshot(findGlobeIframe());
}

/** After pingGlobeForLiveSnapshot — brief wait for postMessage ingest. */
export function waitForGlobeSnapshotUpdate(ms = 2200): Promise<LiveWorldSnapshot | null> {
  return requestLiveSnapshotFromGlobe(findGlobeIframe(), ms);
}
