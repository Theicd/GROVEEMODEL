/** Pose keypoints — MoveNet when reliable, bbox synthesis fallback. */

import type { MotionSnapshot } from "./motionLayer";
import {
  averageKeypointScore,
  estimateMoveNetPose,
  isMoveNetReady,
  preloadMoveNet,
  type PoseSource,
} from "./moveNetLoader";
import type { BBox, Keypoint } from "./poseHeuristics";

export type { PoseSource };

const WRIST_IDX = [9, 10] as const;

export const synthesizeKeypoints = (
  bbox: BBox,
  motion: MotionSnapshot | null,
): Keypoint[] => {
  const kps: Keypoint[] = Array.from({ length: 17 }, () => ({ x: 0, y: 0, score: 0 }));
  const cx = bbox.x + bbox.width / 2;
  const top = bbox.y;
  const h = bbox.height;
  const w = bbox.width;

  const set = (idx: number, x: number, y: number, score = 0.85) => {
    kps[idx] = { x, y, score };
  };

  set(0, cx, top + h * 0.12);
  set(5, cx - w * 0.18, top + h * 0.22);
  set(6, cx + w * 0.18, top + h * 0.22);
  set(11, cx - w * 0.14, top + h * 0.52);
  set(12, cx + w * 0.14, top + h * 0.52);
  set(13, cx - w * 0.12, top + h * 0.72);
  set(14, cx + w * 0.12, top + h * 0.72);

  const armRaised = motion && (motion.waveLike || motion.armMovement || motion.upperMotion > 0.14);
  if (armRaised) {
    set(7, cx - w * 0.28, top + h * 0.18);
    set(9, cx - w * 0.32, top + h * 0.06);
    set(8, cx + w * 0.22, top + h * 0.35);
    set(10, cx + w * 0.24, top + h * 0.42);
  } else {
    set(7, cx - w * 0.26, top + h * 0.38);
    set(8, cx + w * 0.26, top + h * 0.38);
    set(9, cx - w * 0.28, top + h * 0.48);
    set(10, cx + w * 0.28, top + h * 0.48);
  }

  return kps;
};

const mergeWithBbox = (bboxKps: Keypoint[], movenetKps: Keypoint[]): Keypoint[] => {
  const out = bboxKps.map((k) => ({ ...k }));
  for (const idx of WRIST_IDX) {
    if ((movenetKps[idx]?.score ?? 0) >= 0.25) {
      out[idx] = { ...movenetKps[idx] };
    }
  }
  if ((movenetKps[0]?.score ?? 0) >= 0.25) out[0] = { ...movenetKps[0] };
  return out;
};

export type PoseRunResult = {
  keypoints: Keypoint[] | null;
  source: PoseSource;
};

export class PoseDetectorLoop {
  lastKeypoints: Keypoint[] | null = null;
  lastSource: PoseSource = "none";

  reset(): void {
    this.lastKeypoints = null;
    this.lastSource = "none";
  }

  async run(
    video: HTMLVideoElement,
    personBbox: BBox | null,
    motion: MotionSnapshot | null,
  ): Promise<PoseRunResult> {
    const movenetKps = await estimateMoveNetPose(video, personBbox);
    const movenetAvg = movenetKps ? averageKeypointScore(movenetKps) : 0;

    if (personBbox) {
      const bboxKps = synthesizeKeypoints(personBbox, motion);
      if (movenetKps && movenetAvg >= 0.42) {
        this.lastKeypoints = movenetKps;
        this.lastSource = "movenet";
        return { keypoints: movenetKps, source: "movenet" };
      }
      if (movenetKps && movenetAvg >= 0.22) {
        const merged = mergeWithBbox(bboxKps, movenetKps);
        this.lastKeypoints = merged;
        this.lastSource = "merged";
        return { keypoints: merged, source: "merged" };
      }
      this.lastKeypoints = bboxKps;
      this.lastSource = "bbox";
      return { keypoints: bboxKps, source: "bbox" };
    }

    if (movenetKps) {
      this.lastKeypoints = movenetKps;
      this.lastSource = "movenet";
      return { keypoints: movenetKps, source: "movenet" };
    }

    this.lastKeypoints = null;
    this.lastSource = "none";
    return { keypoints: null, source: "none" };
  }
}

export const preloadPoseDetector = preloadMoveNet;
export { isMoveNetReady };
