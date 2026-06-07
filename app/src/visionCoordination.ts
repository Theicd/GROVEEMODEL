/**
 * Avoid TF.js (WebGL) + Gemma (WebGPU) fighting on the same GPU.
 * Main thread sets pause; camera loop reads it before COCO/MoveNet.
 */

let tfVisionPaused = false;
let pauseReason = "";

export const pauseTfVision = (reason = "inference"): void => {
  tfVisionPaused = true;
  pauseReason = reason;
};

export const resumeTfVision = (): void => {
  tfVisionPaused = false;
  pauseReason = "";
};

export const isTfVisionPaused = (): boolean => tfVisionPaused;

export const tfVisionPauseReason = (): string => pauseReason;
