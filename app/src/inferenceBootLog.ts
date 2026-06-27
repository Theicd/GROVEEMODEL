/** Console diagnostics for model load / WebGPU vs WASM — filter DevTools by `GROVEE:boot`. */
const TAG = "[GROVEE:boot]";

export function bootLog(step: string, detail?: Record<string, unknown>): void {
  if (detail !== undefined) {
    console.info(TAG, step, detail);
  } else {
    console.info(TAG, step);
  }
}

export function bootWarn(step: string, detail?: Record<string, unknown>): void {
  if (detail !== undefined) {
    console.warn(TAG, step, detail);
  } else {
    console.warn(TAG, step);
  }
}

export function readWebGpuBlockFlag(): boolean {
  if (typeof localStorage === "undefined") return false;
  try {
    return localStorage.getItem("grovee-webgpu-blocked") === "1";
  } catch {
    return false;
  }
}

export function snapshotInferenceSettings(): Record<string, unknown> {
  if (typeof localStorage === "undefined") return {};
  try {
    const raw = localStorage.getItem("grovee_model_settings_v1");
    const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : null;
    return {
      webgpuBlockedFlag: readWebGpuBlockFlag(),
      inferenceBackend: parsed?.inferenceBackend ?? "(default)",
      localTextBackend:
        parsed && typeof parsed.localText === "object" && parsed.localText !== null
          ? (parsed.localText as Record<string, unknown>).inferenceBackend
          : "(default)",
      startupModel: parsed?.startupModel ?? "(default)",
    };
  } catch {
    return { webgpuBlockedFlag: readWebGpuBlockFlag(), parseError: true };
  }
}
