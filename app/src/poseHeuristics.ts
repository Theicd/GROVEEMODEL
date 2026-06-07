/** COCO-17 / MoveNet keypoint heuristics — ported from JARVIS-VISION store_ai_system. */

export type Keypoint = { x: number; y: number; score: number };

export type PoseState = "standing" | "sitting" | "unknown";

export type PoseInference = {
  poseState: PoseState;
  confidence: number;
};

export type PersonPoseState = {
  poseState: PoseState;
  confidence: number;
  gestures: string[];
  holding: string[];
  focusHint: string;
};

const KP = {
  L_SHOULDER: 5,
  R_SHOULDER: 6,
  L_ELBOW: 7,
  R_ELBOW: 8,
  L_WRIST: 9,
  R_WRIST: 10,
  L_HIP: 11,
  R_HIP: 12,
  L_KNEE: 13,
  R_KNEE: 14,
  NOSE: 0,
} as const;

const MIN_CONF = 0.25;

const kpValid = (kps: Keypoint[], indices: number[]): boolean =>
  indices.every((i) => i < kps.length && (kps[i]?.score ?? 0) >= MIN_CONF);

const torsoHeight = (kps: Keypoint[]): number => {
  const sy = (kps[KP.L_SHOULDER].y + kps[KP.R_SHOULDER].y) / 2;
  const hy = (kps[KP.L_HIP].y + kps[KP.R_HIP].y) / 2;
  return Math.max(1, Math.abs(hy - sy));
};

const poseRequiredIndices = [
  KP.L_SHOULDER,
  KP.R_SHOULDER,
  KP.L_HIP,
  KP.R_HIP,
  KP.L_KNEE,
  KP.R_KNEE,
] as const;

const avgKeypointScore = (kps: Keypoint[], indices: readonly number[]): number => {
  if (!indices.length) return 0;
  return indices.reduce((sum, i) => sum + (kps[i]?.score ?? 0), 0) / indices.length;
};

/** Standing vs sitting from shoulder/hip/knee geometry + confidence score. */
export const inferPoseStateWithConfidence = (kps: Keypoint[]): PoseInference => {
  const required = [...poseRequiredIndices];
  const kpScore = avgKeypointScore(kps, required);
  if (!kpValid(kps, required)) {
    return { poseState: "unknown", confidence: kpScore * 0.35 };
  }

  const shoulderY = (kps[KP.L_SHOULDER].y + kps[KP.R_SHOULDER].y) / 2;
  const hipY = (kps[KP.L_HIP].y + kps[KP.R_HIP].y) / 2;
  const kneeY = (kps[KP.L_KNEE].y + kps[KP.R_KNEE].y) / 2;
  const torso = Math.max(1, hipY - shoulderY);
  const upperLeg = Math.max(0, kneeY - hipY);
  const ratio = upperLeg / torso;

  if (ratio < 0.45) {
    const margin = Math.min(1, (0.45 - ratio) / 0.2);
    return { poseState: "sitting", confidence: Math.min(0.95, (0.5 + margin * 0.45) * kpScore) };
  }
  if (ratio > 0.62) {
    const margin = Math.min(1, (ratio - 0.62) / 0.25);
    return { poseState: "standing", confidence: Math.min(0.95, (0.5 + margin * 0.45) * kpScore) };
  }

  return { poseState: "unknown", confidence: Math.min(0.42, kpScore * 0.4) };
};

export const inferPoseState = (kps: Keypoint[]): PoseState =>
  inferPoseStateWithConfidence(kps).poseState;

/** Wrist raised above shoulders — wave / attention gesture. */
export const detectWaveGesture = (kps: Keypoint[]): boolean => {
  if (!kpValid(kps, [KP.L_WRIST, KP.R_WRIST, KP.L_SHOULDER, KP.R_SHOULDER])) return false;
  const shoulderY = (kps[KP.L_SHOULDER].y + kps[KP.R_SHOULDER].y) / 2;
  const lw = kps[KP.L_WRIST];
  const rw = kps[KP.R_WRIST];
  return lw.y < shoulderY - torsoHeight(kps) * 0.05 || rw.y < shoulderY - torsoHeight(kps) * 0.05;
};

/** Wrist near face level — phone call / thinking pose. */
export const detectPhoneGesture = (kps: Keypoint[], holding: string[]): boolean => {
  if (!holding.some((h) => /phone|cell/.test(h))) return false;
  if (!kpValid(kps, [KP.L_WRIST, KP.R_WRIST, KP.NOSE])) return false;
  const noseY = kps[KP.NOSE].y;
  const th = torsoHeight(kps);
  for (const w of [kps[KP.L_WRIST], kps[KP.R_WRIST]]) {
    if (Math.abs(w.y - noseY) < th * 0.35) return true;
  }
  return false;
};

/** Sitting + laptop/keyboard in holding — focused work. */
export const detectFocusedWork = (
  poseState: string,
  holding: string[],
): boolean => {
  if (poseState !== "sitting") return false;
  return holding.some((h) => /laptop|keyboard|mouse|book/.test(h));
};

export const analyzePersonPose = (
  kps: Keypoint[] | null,
  holding: string[] = [],
): PersonPoseState => {
  if (!kps?.length) {
    return { poseState: "unknown", confidence: 0, gestures: [], holding, focusHint: "" };
  }
  const { poseState, confidence } = inferPoseStateWithConfidence(kps);
  const gestures: string[] = [];
  if (detectWaveGesture(kps)) gestures.push("wave");
  if (detectPhoneGesture(kps, holding)) gestures.push("phone_near_face");
  if (detectFocusedWork(poseState, holding)) gestures.push("focused_work");

  let focusHint = "";
  if (gestures.includes("wave")) focusHint = "raising hand — seeking attention";
  else if (gestures.includes("phone_near_face")) focusHint = "phone interaction";
  else if (gestures.includes("focused_work")) focusHint = "focused on work surface";
  else if (poseState === "sitting") focusHint = "seated, posture relaxed";
  else if (poseState === "standing") focusHint = "upright presence";

  if (holding.some((h) => /cup|bottle/.test(h))) {
    focusHint =
      poseState === "standing" ? "standing with a drink in hand" : "holding a drink";
  }

  return { poseState, confidence, gestures, holding, focusHint };
};

export type BBox = { x: number; y: number; width: number; height: number };

/** Objects near wrists (MoveNet) or expanded person bbox — hand reach margin. */
export const attachHoldingObjects = (
  personBbox: BBox,
  objects: { label: string; bbox: BBox }[],
  marginRatio = 0.2,
  keypoints?: Keypoint[] | null,
): string[] => {
  const held: string[] = [];
  const handRegions = wristHandRegions(keypoints, personBbox);

  for (const obj of objects) {
    const cx = obj.bbox.x + obj.bbox.width / 2;
    const cy = obj.bbox.y + obj.bbox.height / 2;
    const ox2 = obj.bbox.x + obj.bbox.width;
    const oy2 = obj.bbox.y + obj.bbox.height;

    if (handRegions.length) {
      for (const hand of handRegions) {
        if (cx >= hand.x && cx <= hand.x + hand.width && cy >= hand.y && cy <= hand.y + hand.height) {
          held.push(obj.label);
          break;
        }
      }
      continue;
    }

    const mx = personBbox.width * marginRatio;
    const my = personBbox.height * marginRatio;
    const px1 = personBbox.x - mx;
    const py1 = personBbox.y - my;
    const px2 = personBbox.x + personBbox.width + mx;
    const py2 = personBbox.y + personBbox.height + my;
    const centerInside = cx >= px1 && cx <= px2 && cy >= py1 && cy <= py2;
    const overlapsPerson = obj.bbox.x < px2 && ox2 > px1 && obj.bbox.y < py2 && oy2 > py1;
    if (centerInside || overlapsPerson) held.push(obj.label);
  }

  return [...new Set(held)];
};

const wristHandRegions = (kps: Keypoint[] | null | undefined, personBbox: BBox): BBox[] => {
  if (!kps?.length) return [];
  if (!kpValid(kps, [KP.L_WRIST, KP.R_WRIST])) return [];

  const shoulderW = Math.abs(kps[KP.R_SHOULDER].x - kps[KP.L_SHOULDER].x);
  const reach = Math.max(24, shoulderW * 0.35, personBbox.width * 0.12);

  const region = (wrist: Keypoint): BBox => ({
    x: wrist.x - reach,
    y: wrist.y - reach,
    width: reach * 2,
    height: reach * 2,
  });

  return [region(kps[KP.L_WRIST]), region(kps[KP.R_WRIST])];
};
