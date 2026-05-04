import { Txt2ImgWorkerClient } from "web-txt2img";

let client: Txt2ImgWorkerClient | null = null;
let loadPromise: Promise<boolean> | null = null;

export function getSdTurboSizeNote(): string {
  return "~2.3 GB download · 512×512 · WebGPU recommended (WASM fallback may be slow)";
}

function getClient(): Txt2ImgWorkerClient {
  if (!client) client = Txt2ImgWorkerClient.createDefault();
  return client;
}

/** web-txt2img reports webgpu=true when the API exists, not when an adapter is available. */
async function hasWebGpuAdapter(): Promise<boolean> {
  try {
    if (typeof navigator === "undefined" || !navigator.gpu?.requestAdapter) return false;
    const a = await navigator.gpu.requestAdapter({ powerPreference: "low-power" });
    return a != null;
  } catch {
    return false;
  }
}

/**
 * Ensures SD-Turbo weights are cached locally. Safe to call multiple times.
 */
export async function ensureSdTurboLoaded(onStatus: (s: string) => void): Promise<boolean> {
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    const c = getClient();
    const cap = await c.detect();
    if (!cap.wasm) {
      onStatus("Local image: no WASM backend");
      return false;
    }
    const adapterOk = await hasWebGpuAdapter();
    const useWebGpuFirst = !!cap.webgpu && adapterOk;
    onStatus(
      useWebGpuFirst ? "Local image: loading SD-Turbo (WebGPU)…" : "Local image: loading SD-Turbo (WASM/CPU)…",
    );
    const res = await c.load(
      "sd-turbo",
      {
        backendPreference: useWebGpuFirst ? ["webgpu", "wasm"] : ["wasm"],
        onProgress: (p) => {
          const pct = typeof p.pct === "number" ? Math.round(p.pct) : undefined;
          const msg = p.message ?? p.asset ?? "";
          onStatus(pct !== undefined ? `Local image: ${pct}% ${msg}`.trim() : `Local image: ${msg}`.trim());
        },
      },
      (e) => {
        const phase = (e as { phase?: string }).phase ?? "";
        const pct = (e as { pct?: number }).pct;
        if (typeof pct === "number") onStatus(`Local image: ${phase} ${Math.round(pct)}%`);
      },
    );
    if (res && typeof res === "object" && "ok" in res && res.ok === true) {
      onStatus("Local image: SD-Turbo ready");
      return true;
    }
    const reason = res && typeof res === "object" && "message" in res ? String((res as { message?: string }).message) : "";
    onStatus(`Local image failed: ${reason || "load error"}`);
    loadPromise = null;
    return false;
  })();

  const ok = await loadPromise;
  if (!ok) loadPromise = null;
  return ok;
}

export async function generateSdTurboPng(
  englishPrompt: string,
  onStatus: (s: string) => void,
): Promise<{ ok: true; objectUrl: string } | { ok: false; message: string }> {
  const ready = await ensureSdTurboLoaded(onStatus);
  if (!ready) return { ok: false, message: "SD-Turbo not loaded" };

  onStatus("Local image: generating…");
  const c = getClient();
  const { promise, abort } = c.generate(
    { model: "sd-turbo", prompt: englishPrompt, width: 512, height: 512 },
    (e) => {
      const phase = (e as { phase?: string }).phase ?? "";
      onStatus(`Local image: ${phase}`);
    },
  );

  try {
    const msg = await promise;
    // Worker resolves the full message: { type, ok, blob?, timeMs?, reason?, message? }
    if (msg && typeof msg === "object" && "ok" in msg && msg.ok === true && "blob" in msg) {
      const blob = (msg as { blob: Blob }).blob;
      const objectUrl = URL.createObjectURL(blob);
      onStatus("Local image: done");
      return { ok: true, objectUrl };
    }
    const errText =
      msg && typeof msg === "object" && "message" in msg
        ? String((msg as { message?: string }).message)
        : "generation failed";
    return { ok: false, message: errText };
  } catch (e) {
    void abort();
    const message = e instanceof Error ? e.message : String(e);
    return { ok: false, message };
  }
}

export function revokeImageUrl(url: string | null) {
  if (url && url.startsWith("blob:")) {
    try {
      URL.revokeObjectURL(url);
    } catch {
      // ignore
    }
  }
}

export function terminateLocalImageWorker() {
  client?.terminate();
  client = null;
  loadPromise = null;
}
