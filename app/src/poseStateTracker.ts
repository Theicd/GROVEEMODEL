/** Multi-frame pose validation — commit only after consecutive agreement. */

import type { PoseInference } from "./poseHeuristics";

export type PoseTrackResult = {
  committed: boolean;
  changed: boolean;
  from: "standing" | "sitting" | "unknown";
  to: "standing" | "sitting" | "unknown";
  confidence: number;
};

export const POSE_TRACKER_CONFIG = {
  framesRequired: 2,
  minConfidence: 0.45,
  historySize: 5,
} as const;

export class PoseStateTracker {
  private history: PoseInference[] = [];
  committedState: "standing" | "sitting" | "unknown" = "unknown";
  committedConfidence = 0;

  reset(): void {
    this.history = [];
    this.committedState = "unknown";
    this.committedConfidence = 0;
  }

  observe(inference: PoseInference): PoseTrackResult {
    this.history.push(inference);
    if (this.history.length > POSE_TRACKER_CONFIG.historySize) {
      this.history.shift();
    }

    const from = this.committedState;
    const recent = this.history.slice(-POSE_TRACKER_CONFIG.framesRequired);
    if (recent.length < POSE_TRACKER_CONFIG.framesRequired) {
      return { committed: false, changed: false, from, to: from, confidence: inference.confidence };
    }

    const state = recent[0].poseState;
    const allAgree =
      state !== "unknown" && recent.every((r) => r.poseState === state);
    const avgConf =
      recent.reduce((sum, r) => sum + r.confidence, 0) / recent.length;

    if (!allAgree || avgConf < POSE_TRACKER_CONFIG.minConfidence) {
      return { committed: false, changed: false, from, to: from, confidence: avgConf };
    }

    const changed = from !== "unknown" && from !== state;
    this.committedState = state;
    this.committedConfidence = avgConf;

    return {
      committed: true,
      changed,
      from,
      to: state,
      confidence: avgConf,
    };
  }

  /** For chat refresh — require two consecutive observations outside normal loop. */
  mergeObservations(observations: PoseInference[]): PoseInference {
    if (!observations.length) {
      return { poseState: "unknown", confidence: 0 };
    }
    if (observations.length === 1) return observations[0];

    const states = observations.map((o) => o.poseState);
    const avgConf =
      observations.reduce((s, o) => s + o.confidence, 0) / observations.length;
    const allSame = states.every((s) => s === states[0] && s !== "unknown");

    if (allSame) {
      return { poseState: states[0], confidence: Math.min(0.98, avgConf + 0.08) };
    }
    if (avgConf >= POSE_TRACKER_CONFIG.minConfidence) {
      const standing = states.filter((s) => s === "standing").length;
      const sitting = states.filter((s) => s === "sitting").length;
      if (standing > sitting) {
        return { poseState: "standing", confidence: avgConf * 0.75 };
      }
      if (sitting > standing) {
        return { poseState: "sitting", confidence: avgConf * 0.75 };
      }
    }
    return { poseState: "unknown", confidence: avgConf * 0.5 };
  }
}
