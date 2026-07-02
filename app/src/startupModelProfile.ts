import {
  SMOLLM_135M_RACK_ID,
  SMOLLM_RACK_ID,
} from "./modelRack/localTextModels";

/** SmolLM2 360M Instruct ONNX q4 — approximate browser download size. */
export const SMOLLM_ESTIMATED_BYTES = 220_000_000;

/** SmolLM2 135M — lighter fallback. */
export const SMOLLM_135M_ESTIMATED_BYTES = 175_000_000;

export type StartupModelChoice = "gemma" | "local-text";

export type StartupModelPreference = "auto" | StartupModelChoice;

export type WebGpuAdapterProbe = {
  available: boolean;
  isFallbackAdapter: boolean;
  vendor: string;
  architecture: string;
  description: string;
};

export type StartupDeviceSignals = {
  deviceMemoryGb: number | null;
  hardwareConcurrency: number | null;
  isMobile: boolean;
  webgpu: WebGpuAdapterProbe;
};

export type StartupModelRecommendation = {
  choice: StartupModelChoice;
  reasonHe: string;
  signals: StartupDeviceSignals;
  fromPreference: boolean;
};

export async function probeWebGpuAdapter(): Promise<WebGpuAdapterProbe> {
  const empty: WebGpuAdapterProbe = {
    available: false,
    isFallbackAdapter: false,
    vendor: "",
    architecture: "",
    description: "",
  };
  if (typeof navigator === "undefined") return empty;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) {
    console.info("[GROVEE:boot]", "probeWebGpuAdapter: navigator.gpu missing");
    return empty;
  }
  try {
    const adapter = (await requestAdapterWithTimeout(() => gpu.requestAdapter())) as {
      info?: {
        isFallbackAdapter?: boolean;
        vendor?: string;
        architecture?: string;
        description?: string;
      };
    } | null;
    if (!adapter) {
      console.info("[GROVEE:boot]", "probeWebGpuAdapter: requestAdapter() returned null (or timed out)");
      return empty;
    }
    const info = adapter.info;
    const result = {
      available: true,
      isFallbackAdapter: !!info?.isFallbackAdapter,
      vendor: info?.vendor ?? "",
      architecture: info?.architecture ?? "",
      description: info?.description ?? "",
    };
    console.info("[GROVEE:boot]", "probeWebGpuAdapter: OK", result);
    return result;
  } catch (err) {
    console.warn("[GROVEE:boot]", "probeWebGpuAdapter: threw", {
      error: err instanceof Error ? err.message : String(err),
    });
    return empty;
  }
}

export function detectMobileDevice(): boolean {
  if (typeof window === "undefined" || typeof navigator === "undefined") return false;
  const ua = navigator.userAgent;
  if (/Android|iPhone|iPad|iPod|Mobile|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua)) {
    return true;
  }
  const coarse = window.matchMedia?.("(pointer: coarse)")?.matches ?? false;
  const narrow = window.innerWidth < 900;
  return coarse && narrow;
}

const WEBGPU_PROBE_TIMEOUT_MS = 2_500;

async function requestAdapterWithTimeout(
  request: () => Promise<unknown | null>,
): Promise<unknown | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      request(),
      new Promise<null>((resolve) => {
        timer = setTimeout(() => resolve(null), WEBGPU_PROBE_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** SmolLM on phones: WASM first — WebGPU probe often stalls before any download bytes. */
export function resolveLocalTextBootBackend(
  inferenceBackend: "auto" | "webgpu" | "wasm",
  opts?: { forceWasm?: boolean },
): "auto" | "webgpu" | "wasm" {
  if (opts?.forceWasm) return "wasm";
  if (detectMobileDevice()) return "wasm";
  return inferenceBackend;
}

export async function collectStartupDeviceSignals(): Promise<StartupDeviceSignals> {
  const nav = typeof navigator !== "undefined" ? navigator : null;
  const mem = nav ? (nav as Navigator & { deviceMemory?: number }).deviceMemory : undefined;
  const webgpu = await probeWebGpuAdapter();
  return {
    deviceMemoryGb: typeof mem === "number" && Number.isFinite(mem) ? mem : null,
    hardwareConcurrency:
      nav && typeof nav.hardwareConcurrency === "number" ? nav.hardwareConcurrency : null,
    isMobile: detectMobileDevice(),
    webgpu,
  };
}

export function recommendStartupModel(signals: StartupDeviceSignals): StartupModelRecommendation {
  const reasons: string[] = [];

  if (signals.isMobile) reasons.push("מכשיר נייד");
  if (signals.deviceMemoryGb != null && signals.deviceMemoryGb <= 4) {
    reasons.push(`זיכרון מוגבל (~${signals.deviceMemoryGb}GB)`);
  }
  if (!signals.webgpu.available) reasons.push("ללא WebGPU");
  if (signals.webgpu.isFallbackAdapter) reasons.push("GPU תוכנה (fallback)");

  const preferLocal =
    signals.isMobile ||
    (signals.deviceMemoryGb != null && signals.deviceMemoryGb <= 4) ||
    !signals.webgpu.available ||
    signals.webgpu.isFallbackAdapter;

  return {
    choice: preferLocal ? "local-text" : "gemma",
    reasonHe: preferLocal
      ? reasons.length
        ? reasons.join(" · ")
        : "מכשיר חלש"
      : "מחשב חזק · WebGPU זמין",
    signals,
    fromPreference: false,
  };
}

export function quickStartupModelChoice(
  preference: StartupModelPreference = "auto",
): StartupModelChoice {
  if (preference === "local-text" || preference === "gemma") return preference;
  return detectMobileDevice() ? "local-text" : "gemma";
}

export async function resolveStartupModelChoice(
  preference: StartupModelPreference = "auto",
): Promise<StartupModelRecommendation> {
  const signals = await collectStartupDeviceSignals();
  if (preference === "gemma" || preference === "local-text") {
    return {
      choice: preference,
      reasonHe: preference === "local-text" ? "הגדרה: SmolLM בפתיחה" : "הגדרה: Gemma בפתיחה",
      signals,
      fromPreference: true,
    };
  }
  return recommendStartupModel(signals);
}

/** Pick 135M on low memory; 360M otherwise for local-text boot. */
export function recommendLocalTextRackId(signals: StartupDeviceSignals): string {
  const lowMem = signals.deviceMemoryGb != null && signals.deviceMemoryGb <= 4;
  if (lowMem) return SMOLLM_135M_RACK_ID;
  if (signals.isMobile && signals.deviceMemoryGb != null && signals.deviceMemoryGb <= 6) {
    return SMOLLM_135M_RACK_ID;
  }
  return SMOLLM_RACK_ID;
}

export function localTextRackLabelHe(rackId: string): string {
  if (rackId === SMOLLM_135M_RACK_ID) return "SmolLM2 135M";
  if (rackId === SMOLLM_RACK_ID) return "SmolLM2 360M";
  return "SmolLM2";
}

export function startupChoiceLabelHe(
  choice: StartupModelChoice,
  localTextRackId?: string,
): string {
  if (choice === "local-text") {
    return localTextRackId ? localTextRackLabelHe(localTextRackId) : "SmolLM2 360M";
  }
  return "Gemma 4 E2B";
}
