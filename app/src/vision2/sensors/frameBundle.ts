/**
 * L1 private bundle — vision frame + optional audio.
 * Never serialized to DialogueContext / LLM.
 */

import type { VisionResult } from "../../vision-lab/core/types";
import type { AudioSample } from "./audioSensor";

export type FrameBundle = {
  timestamp: number;
  vision: VisionResult;
  audio: AudioSample | null;
};

export const createFrameBundle = (
  vision: VisionResult,
  audio: AudioSample | null,
  now = Date.now(),
): FrameBundle => ({
  timestamp: now,
  vision,
  audio,
});
