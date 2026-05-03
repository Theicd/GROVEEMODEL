import type { ModelId } from "web-txt2img";
import { Txt2ImgWorkerClient } from "web-txt2img";

let client: Txt2ImgWorkerClient | null = null;
let loadPromise: Promise<boolean> | null = null;
let loadedModelId: ModelId | null = null;

export type ImageGenModelId = ModelId;

/**
 * Options passed to `Txt2ImgWorkerClient.load` must be structured-cloneable — the library
 * posts `{ model, options }` to a Worker. Never put callbacks here (use the 3rd `onProgress` arg).
 */
export function webTxt2ImgLoadOptions(
  modelId: ImageGenModelId,
  useWebGpuFirst: boolean,
): {
  backendPreference: Array<"webgpu" | "wasm">;
} {
  if (modelId === "janus-pro-1b") {
    return { backendPreference: ["webgpu"] };
  }
  return {
    backendPreference: useWebGpuFirst ? ["webgpu", "wasm"] : ["wasm"],
  };
}

export function getImageGenModelSizeNote(modelId: ImageGenModelId): string {
  if (modelId === "janus-pro-1b") {
    return "Janus-Pro-1B: WebGPU only; large ONNX bundle — first run downloads model assets.";
  }
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
    const a = await navigator.gpu.requestAdapter();
    return a != null;
  } catch {
    return false;
  }
}

function resetImageClient() {
  client?.terminate();
  client = null;
  loadPromise = null;
  loadedModelId = null;
}

/**
 * Ensures the selected web-txt2img model weights are cached locally. Safe to call multiple times.
 * If `modelId` changes, the worker client is reset and the new model is loaded.
 */
export async function ensureLocalImageModelLoaded(
  modelId: ImageGenModelId,
  onStatus: (s: string) => void,
  onStagePct?: (pct0to100: number) => void,
): Promise<boolean> {
  if (loadedModelId !== null && loadedModelId !== modelId) {
    resetImageClient();
  }

  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    const c = getClient();
    const cap = await c.detect();
    if (!cap.wasm) {
      onStatus("Local image: no WASM backend");
      return false;
    }
    const adapterOk = await hasWebGpuAdapter();
    const useWebGpuFirst = modelId === "sd-turbo" && !!cap.webgpu && adapterOk;
    if (modelId === "janus-pro-1b" && !adapterOk) {
      onStatus("Local image: Janus-Pro requires WebGPU with a working adapter");
      loadPromise = null;
      return false;
    }
    onStatus(
      modelId === "janus-pro-1b"
        ? "Local image: loading Janus-Pro-1B (WebGPU)…"
        : useWebGpuFirst
          ? "Local image: loading SD-Turbo (WebGPU)…"
          : "Local image: loading SD-Turbo (WASM/CPU)…",
    );
    const res = await c.load(modelId, webTxt2ImgLoadOptions(modelId, useWebGpuFirst), (p) => {
      const raw = p as { phase?: string; pct?: number; message?: string; asset?: string };
      const phase = raw.phase ?? "";
      const pct = typeof raw.pct === "number" ? Math.round(raw.pct) : undefined;
      const msg = raw.message ?? raw.asset ?? "";
      const bits: string[] = [];
      if (phase) bits.push(phase);
      if (pct !== undefined) bits.push(`${pct}%`);
      if (msg) bits.push(msg);
      onStatus(`Local image: ${bits.length ? bits.join(" · ") : "…"}`);
      if (pct !== undefined) onStagePct?.(pct);
    });
    if (res && typeof res === "object" && "ok" in res && res.ok === true) {
      loadedModelId = modelId;
      onStatus(`Local image: ${modelId} ready`);
      onStagePct?.(100);
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

export async function generateLocalImagePng(
  modelId: ImageGenModelId,
  englishPrompt: string,
  onStatus: (s: string) => void,
): Promise<{ ok: true; objectUrl: string } | { ok: false; message: string }> {
  const ready = await ensureLocalImageModelLoaded(modelId, onStatus);
  if (!ready) return { ok: false, message: `${modelId} not loaded` };

  onStatus(`Local image: generating (${modelId})…`);
  const c = getClient();
  const genParams =
    modelId === "janus-pro-1b"
      ? { model: modelId, prompt: englishPrompt }
      : { model: modelId, prompt: englishPrompt, width: 512, height: 512 };
  const { promise, abort } = c.generate(
    genParams,
    (e) => {
      const phase = (e as { phase?: string }).phase ?? "";
      onStatus(`Local image: ${phase}`);
    },
  );

  try {
    const msg = await promise;
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
  resetImageClient();
}
