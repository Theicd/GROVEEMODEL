import type { GroveeVisionRunner } from "./GroveeVisionRunner";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";

export type GroveeVisionProbeSnapshot = {
  ts: number;
  latest: VisionResult | null;
  config: PipelineConfig | null;
  pipelinePaused: boolean;
  analyzing: boolean;
  deepBaselineDone: boolean;
  deepVisionDegraded: boolean;
  faceOk: boolean;
  emotionOk: boolean;
};

export type GroveeVisionProbe = {
  snapshot: () => GroveeVisionProbeSnapshot;
  waitForFaceData: (timeoutMs?: number) => Promise<GroveeVisionProbeSnapshot>;
};

export const createGroveeVisionProbe = (runner: GroveeVisionRunner | null): GroveeVisionProbe => {
  const snapshot = (): GroveeVisionProbeSnapshot => {
    const latest = runner?.getLatestResult() ?? null;
    const config = runner?.getPipeline().getConfig() ?? null;
    const face = latest?.faces?.[0];
    const faceOk = !!face && face.estimatedAge > 0 && !!face.estimatedGender;
    const emotionOk =
      !!latest?.emotion &&
      latest.emotion.dominantScore > 0 &&
      latest.faceModule.status === "ready";
    return {
      ts: Date.now(),
      latest,
      config,
      pipelinePaused: runner?.isPipelinePaused() ?? false,
      analyzing: runner?.isAnalyzing() ?? false,
      deepBaselineDone: runner?.isDeepBaselineDone() ?? false,
      deepVisionDegraded: runner?.isDeepVisionDegraded() ?? false,
      faceOk,
      emotionOk,
    };
  };

  const waitForFaceData = async (timeoutMs = 90_000): Promise<GroveeVisionProbeSnapshot> => {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const snap = snapshot();
      if (snap.faceOk && snap.emotionOk) return snap;
      if (snap.latest?.faceModule.status === "error") return snap;
      await new Promise((r) => setTimeout(r, 500));
    }
    return snapshot();
  };

  return { snapshot, waitForFaceData };
};

declare global {
  interface Window {
    __groveeVisionProbe?: GroveeVisionProbe;
  }
}

export const mountGroveeVisionProbe = (runner: GroveeVisionRunner | null): void => {
  if (!import.meta.env.DEV) return;
  window.__groveeVisionProbe = createGroveeVisionProbe(runner);
};
