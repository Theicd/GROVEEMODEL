/**
 * Adaptive vision budget for static hosting (GitHub Pages) — balance HAL UX vs device load.
 * Stage 1–2 (motion, COCO, heuristics) stay on; Stage 3 (Gemma vision) is throttled on weak devices.
 */

export type VisionBudgetTier = "low" | "normal";

export type VisionBudgetProfile = {
  tier: VisionBudgetTier;
  /** Why this tier was chosen (for activity log / status). */
  reason: string;
  pollIntervalMs: number;
  deepVisionMinIntervalMs: number;
  deepVisionMaxIntervalMs: number;
  /** Skip MoveNet preload on tier low — saves TF.js work when alone. */
  preloadPose: boolean;
  /** Allow Gemma analyze_scene (deep world memory). */
  useLlmDeepVision: boolean;
  /** Polish proactive lines with Gemma; fallback Hebrew still works when false. */
  useLlmProactiveUtterance: boolean;
  /** After this many consecutive deep-vision failures, stop retrying for a while. */
  maxDeepVisionFailures: number;
  baseDeepVisionBackoffMs: number;
  /** Reject tiny/black JPEG captures (bytes). */
  minSnapshotBytes: number;
};

const NORMAL: VisionBudgetProfile = {
  tier: "normal",
  reason: "default",
  pollIntervalMs: 4000,
  deepVisionMinIntervalMs: 120_000,
  deepVisionMaxIntervalMs: 300_000,
  preloadPose: true,
  useLlmDeepVision: true,
  useLlmProactiveUtterance: false,
  maxDeepVisionFailures: 3,
  baseDeepVisionBackoffMs: 90_000,
  minSnapshotBytes: 8000,
};

const LOW: VisionBudgetProfile = {
  tier: "low",
  reason: "constrained device",
  pollIntervalMs: 6000,
  deepVisionMinIntervalMs: 300_000,
  deepVisionMaxIntervalMs: 600_000,
  preloadPose: false,
  useLlmDeepVision: false,
  useLlmProactiveUtterance: false,
  maxDeepVisionFailures: 2,
  baseDeepVisionBackoffMs: 180_000,
  minSnapshotBytes: 8000,
};

/** Heuristic device tier — no server; runs in browser only. */
export const detectVisionBudget = (): VisionBudgetProfile => {
  if (typeof navigator === "undefined") return { ...NORMAL };

  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const cores = navigator.hardwareConcurrency ?? 4;
  const reducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true;

  if (reducedMotion) {
    return { ...LOW, reason: "prefers-reduced-motion" };
  }
  if (typeof mem === "number" && mem <= 4) {
    return { ...LOW, reason: `deviceMemory<=${mem}GB` };
  }
  if (cores <= 4) {
    return { ...LOW, reason: `hardwareConcurrency<=${cores}` };
  }

  return { ...NORMAL };
};

export const deepVisionBackoffMs = (
  profile: VisionBudgetProfile,
  failureCount: number,
): number => {
  if (failureCount <= 0) return 0;
  const mult = Math.min(4, failureCount);
  return profile.baseDeepVisionBackoffMs * mult;
};

export const mergeCameraLoopTiming = <T extends {
  pollIntervalMs: number;
  deepVisionMinIntervalMs: number;
  deepVisionMaxIntervalMs: number;
  minFrameStabilityForDeep: number;
}>(
  base: T,
  profile: VisionBudgetProfile,
): T => ({
  ...base,
  pollIntervalMs: profile.pollIntervalMs,
  deepVisionMinIntervalMs: profile.deepVisionMinIntervalMs,
  deepVisionMaxIntervalMs: profile.deepVisionMaxIntervalMs,
});
