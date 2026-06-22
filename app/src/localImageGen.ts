import { Txt2ImgWorkerClient } from "web-txt2img";

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

export const SD_TURBO_UNAVAILABLE_MSG =
  "SD-Turbo מקומי דורש web-txt2img — בחר FLUX (Pollinations) או התקן את החבילה";

export function getSdTurboSizeNote(): string {
  return "~2.3 GB download · 512×512 · WebGPU recommended (WASM fallback may be slow)";
}

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

async function hasWebGpuAdapter(): Promise<boolean> {
  try {
    if (typeof navigator === "undefined" || !navigator.gpu?.requestAdapter) return false;
    const a = await navigator.gpu.requestAdapter({ powerPreference: "low-power" });
    return a != null;
  } catch {
    return false;
  }
}

export function buildLoadOptions(useWebGpuFirst: boolean): { backendPreference: ("webgpu" | "wasm")[] } {
  return { backendPreference: useWebGpuFirst ? ["webgpu", "wasm"] : ["wasm"] };
}

export function buildGenerateParams(prompt: string): {
  model: "sd-turbo";
  prompt: string;
  width: number;
  height: number;
} {
  return { model: "sd-turbo", prompt, width: 512, height: 512 };
}

export type SdTurboBackendPref = "auto" | "webgpu" | "wasm";

export interface EnsureSdTurboLoadedOptions {
  forceBackend?: SdTurboBackendPref;
}

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
      onStatus(`Local image: ${SD_TURBO_UNAVAILABLE_MSG}`);
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
    onStatus(`Local image failed: ${reason || SD_TURBO_UNAVAILABLE_MSG}`);
    loadPromise = null;
    return false;
  })();

  const ok = await loadPromise;
  if (!ok) loadPromise = null;
  return ok;
}

export function isWebGpuStateError(message: string | null | undefined): boolean {
  if (!message) return false;
  return /reading\s+'destroy'|external Instance|GPU device was lost|Aborted\(.*\)/i.test(message);
}

export async function generateSdTurboPng(
  englishPrompt: string,
  onStatus: (s: string) => void,
): Promise<{ ok: true; objectUrl: string } | { ok: false; message: string }> {
  const ready = await ensureSdTurboLoaded(onStatus);
  if (!ready) return { ok: false, message: SD_TURBO_UNAVAILABLE_MSG };

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
        : SD_TURBO_UNAVAILABLE_MSG;
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
      /* ignore */
    }
  }
}

export function terminateLocalImageWorker() {
  client?.terminate();
  client = null;
  loadPromise = null;
}
