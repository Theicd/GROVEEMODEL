/**
 * Vision / HAL behavior settings persisted with app settings.
 */

import type { PerformanceMode } from "./vision-lab/core/types";

export type VisionBehaviorSettings = {
  /** Gemma analyze_scene on boot + significant changes */
  useLlmDeepVision: boolean;
  /** Polish proactive Hebrew lines with Gemma (costly) */
  useLlmProactiveUtterance: boolean;
  /** Run deep snapshot once when camera starts */
  useBootDeepSnapshot: boolean;
  /** Default vision-lab performance preset */
  performanceMode: PerformanceMode;
  /** Show full detection cards in Vision Inspector */
  showDetectionCards: boolean;
  /** Log vision sync to activity panel */
  logVisionToActivity: boolean;
};

export const DEFAULT_VISION_SETTINGS: VisionBehaviorSettings = {
  useLlmDeepVision: true,
  useLlmProactiveUtterance: false,
  useBootDeepSnapshot: true,
  performanceMode: "balanced",
  showDetectionCards: true,
  logVisionToActivity: true,
};

export const mergeVisionSettings = (partial?: Partial<VisionBehaviorSettings>): VisionBehaviorSettings => ({
  ...DEFAULT_VISION_SETTINGS,
  ...partial,
});
