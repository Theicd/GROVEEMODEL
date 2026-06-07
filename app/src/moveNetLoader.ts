/** Lazy-loaded MoveNet (TF.js) — separate chunk, bbox fallback if load fails. */

import { ensureTfBackend } from "./browserVision";
import type { BBox, Keypoint } from "./poseHeuristics";

export type PoseSource = "movenet" | "bbox" | "merged" | "vision-lab" | "none";

type MoveNetDetector = {
  estimatePoses: (
    input: HTMLVideoElement | HTMLCanvasElement | ImageBitmap,
    config?: { maxPoses?: number; flipHorizontal?: boolean },
  ) => Promise<Array<{ keypoints: Array<{ x: number; y: number; score?: number }> }>>;
};

let detectorPromise: Promise<MoveNetDetector | null> | null = null;
let moveNetReady = false;
let loadFailed = false;

export const isMoveNetReady = (): boolean => moveNetReady;

export const movenetKeypointsToCoco = (
  raw: Array<{ x: number; y: number; score?: number }>,
): Keypoint[] => {
  const kps: Keypoint[] = Array.from({ length: 17 }, () => ({ x: 0, y: 0, score: 0 }));
  for (let i = 0; i < Math.min(17, raw.length); i++) {
    kps[i] = { x: raw[i].x, y: raw[i].y, score: raw[i].score ?? 0 };
  }
  return kps;
};

export const averageKeypointScore = (kps: Keypoint[]): number => {
  if (!kps.length) return 0;
  return kps.reduce((s, k) => s + k.score, 0) / kps.length;
};

const loadDetector = async (): Promise<MoveNetDetector | null> => {
  if (loadFailed) return null;
  try {
    await ensureTfBackend(true);
    const [{ load }, { SINGLEPOSE_LIGHTNING }] = await Promise.all([
      import("@tensorflow-models/pose-detection/dist/movenet/detector"),
      import("@tensorflow-models/pose-detection/dist/movenet/constants"),
    ]);
    const detector = await load({
      modelType: SINGLEPOSE_LIGHTNING,
      enableSmoothing: true,
    });
    moveNetReady = true;
    return detector;
  } catch (e) {
    console.warn("[GROVEE] MoveNet load failed — bbox pose fallback", e);
    loadFailed = true;
    moveNetReady = false;
    return null;
  }
};

export const preloadMoveNet = async (): Promise<boolean> => {
  if (!detectorPromise) detectorPromise = loadDetector();
  return (await detectorPromise) !== null;
};

export const estimateMoveNetPose = async (
  video: HTMLVideoElement,
  personBbox?: BBox | null,
): Promise<Keypoint[] | null> => {
  if (video.readyState < 2 || video.videoWidth <= 0) return null;
  if (!detectorPromise) detectorPromise = loadDetector();
  const detector = await detectorPromise;
  if (!detector) return null;

  try {
    const poses = await detector.estimatePoses(video, { maxPoses: 1, flipHorizontal: false });
    const raw = poses[0]?.keypoints;
    if (!raw?.length) return null;

    const kps = movenetKeypointsToCoco(raw);
    const avg = averageKeypointScore(kps);

    if (avg < 0.22) return null;

    if (personBbox && avg < 0.42) {
      const inBox = kps.filter(
        (k) =>
          k.score > 0.15 &&
          k.x >= personBbox.x &&
          k.x <= personBbox.x + personBbox.width &&
          k.y >= personBbox.y &&
          k.y <= personBbox.y + personBbox.height,
      ).length;
      if (inBox < 4) return null;
    }

    return kps;
  } catch (e) {
    console.warn("[GROVEE] MoveNet estimate failed", e);
    return null;
  }
};

export const resetMoveNetLoader = (): void => {
  detectorPromise = null;
  moveNetReady = false;
  loadFailed = false;
};
