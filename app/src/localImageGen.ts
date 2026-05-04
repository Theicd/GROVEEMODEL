import { Txt2ImgWorkerClient } from "web-txt2img";

/**
 * Public type used by the load/generate adapters so tests can run without `web-txt2img`.
 * Keep aligned with `web-txt2img/dist/types.d.ts` `LoadProgress`.
 */
export type LoadProgressLike = {
  phase?: string;
  message?: string;
  pct?: number;
  asset?: string;
};

export type GenProgressLike = {
  phase?: string;
  pct?: number;
};

/**
 * Subset of `Txt2ImgWorkerClient` we actually call. Keeps tests honest about the contract:
 * - `load(model, options, onProgress)` — options is structured-clone safe (NO functions),
 *   onProgress is the third arg and stays on the main thread.
 */
export interface ImageWorkerClientLike {
  detect(): Promise<{ webgpu?: boolean; wasm?: boolean; shaderF16?: boolean }>;
  load(
    model: string,
    options: { backendPreference: ("webgpu" | "wasm")[] },
    onProgress?: (p: LoadProgressLike) => void,
  ): Promise<{ ok: boolean; reason?: string; message?: string } | unknown>;
  generate(
    params: { model: string; prompt: string; width?: number; height?: number; seed?: number },
    onProgress?: (e: GenProgressLike) => void,
  ): { id: string; promise: Promise<unknown>; abort: () => Promise<void> };
  terminate(): void;
}

let client: ImageWorkerClientLike | null = null;
let loadPromise: Promise<boolean> | null = null;

export function getSdTurboSizeNote(): string {
  return "~2.3 GB download · 512×512 · WebGPU recommended (WASM fallback may be slow)";
}

/**
 * Test seam: tests can replace the client with a mock. In production we lazily
 * build the real `Txt2ImgWorkerClient.createDefault()`.
 */
let clientFactory: () => ImageWorkerClientLike = () =>
  Txt2ImgWorkerClient.createDefault() as unknown as ImageWorkerClientLike;

export function __setImageClientFactoryForTests(factory: (() => ImageWorkerClientLike) | null): void {
  clientFactory = factory ?? (() => Txt2ImgWorkerClient.createDefault() as unknown as ImageWorkerClientLike);
  client = null;
  loadPromise = null;
}

function getClient(): ImageWorkerClientLike {
  if (!client) client = clientFactory();
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
 * Build the load options that get postMessage'd to the worker. MUST be JSON-serializable
 * (no functions / no AbortSignal). Exported for tests.
 */
export function buildLoadOptions(useWebGpuFirst: boolean): { backendPreference: ("webgpu" | "wasm")[] } {
  return { backendPreference: useWebGpuFirst ? ["webgpu", "wasm"] : ["wasm"] };
}

/**
 * Build the generate params. MUST be JSON-serializable (no functions / no AbortSignal).
 * Exported for tests.
 */
export function buildGenerateParams(prompt: string): {
  model: "sd-turbo";
  prompt: string;
  width: number;
  height: number;
} {
  return { model: "sd-turbo", prompt, width: 512, height: 512 };
}

export type SdTurboBackendPref = "auto" | "webgpu" | "wasm";

/**
 * Loader signature. Exported so App can pass `forceBackend: "wasm"` for the
 * documented retry path when WebGPU dies mid-init (the
 * `Cannot read properties of undefined (reading 'destroy')` /
 * `A valid external Instance reference no longer exists` family of errors,
 * very common with multiple ONNX sessions sharing one GPU adapter).
 */
export interface EnsureSdTurboLoadedOptions {
  forceBackend?: SdTurboBackendPref;
}

/** Ensures SD-Turbo weights are cached locally. Safe to call multiple times. */
export async function ensureSdTurboLoaded(
  onStatus: (s: string) => void,
  options: EnsureSdTurboLoadedOptions = {},
): Promise<boolean> {
  if (loadPromise) return loadPromise;
  const forceBackend = options.forceBackend ?? "auto";

  loadPromise = (async () => {
    const c = getClient();
    const cap = await c.detect();
    if (!cap.wasm) {
      onStatus("Local image: no WASM backend");
      return false;
    }
    const adapterOk = forceBackend === "wasm" ? false : await hasWebGpuAdapter();
    const useWebGpuFirst =
      forceBackend === "wasm" ? false : forceBackend === "webgpu" ? !!cap.webgpu && adapterOk : !!cap.webgpu && adapterOk;
    onStatus(
      useWebGpuFirst
        ? "Local image: loading SD-Turbo (WebGPU)…"
        : forceBackend === "wasm"
          ? "Local image: loading SD-Turbo (WASM/CPU, forced)…"
          : "Local image: loading SD-Turbo (WASM/CPU)…",
    );
    const res = await c.load("sd-turbo", buildLoadOptions(useWebGpuFirst), (p: LoadProgressLike) => {
      const pct = typeof p.pct === "number" ? Math.round(p.pct) : undefined;
      const msg = p.message ?? p.asset ?? "";
      onStatus(pct !== undefined ? `Local image: ${pct}% ${msg}`.trim() : `Local image: ${msg}`.trim());
    });
    if (res && typeof res === "object" && "ok" in res && (res as { ok: boolean }).ok === true) {
      onStatus("Local image: SD-Turbo ready");
      return true;
    }
    const reason =
      res && typeof res === "object" && "message" in res ? String((res as { message?: string }).message) : "";
    onStatus(`Local image failed: ${reason || "load error"}`);
    loadPromise = null;
    return false;
  })();

  const ok = await loadPromise;
  if (!ok) loadPromise = null;
  return ok;
}

/**
 * Heuristic for the GPU-state errors that show up under memory pressure when
 * multiple ONNX sessions share one adapter. Used by App to decide whether to
 * retry SD-Turbo on WASM after a WebGPU failure.
 */
export function isWebGpuStateError(message: string | null | undefined): boolean {
  if (!message) return false;
  return /reading\s+'destroy'|external Instance|GPU device was lost|Aborted\(.*\)/i.test(message);
}

export async function generateSdTurboPng(
  englishPrompt: string,
  onStatus: (s: string) => void,
): Promise<{ ok: true; objectUrl: string } | { ok: false; message: string }> {
  const ready = await ensureSdTurboLoaded(onStatus);
  if (!ready) return { ok: false, message: "SD-Turbo not loaded" };

  onStatus("Local image: generating…");
  const c = getClient();
  const { promise, abort } = c.generate(buildGenerateParams(englishPrompt), (e: GenProgressLike) => {
    const phase = e.phase ?? "";
    const pct = e.pct;
    onStatus(typeof pct === "number" ? `Local image: ${phase} ${Math.round(pct)}%` : `Local image: ${phase}`);
  });

  try {
    const msg = await promise;
    if (msg && typeof msg === "object" && "ok" in msg && (msg as { ok: boolean }).ok === true && "blob" in msg) {
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
