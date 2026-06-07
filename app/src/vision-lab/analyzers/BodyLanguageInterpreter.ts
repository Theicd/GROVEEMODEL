import type {
  BodyLanguageCue,
  DetectedFace,
  DetectedHand,
  MotionGesture,
  PoseAction,
  PoseLandmark,
  StaticGesture,
} from '../core/types';
import { bboxCenter, distance } from '../utils/geometry';

const STATIC_SIGN_MAP: Record<string, { meaning: string; confidence: number }> = {
  'Thumbs Up': { meaning: 'Like / OK / Approval', confidence: 0.9 },
  'Thumbs Down': { meaning: 'Dislike / Not OK / Rejection', confidence: 0.9 },
  'OK Sign': { meaning: 'OK / All good', confidence: 0.88 },
  'Peace Sign': { meaning: 'Peace / Victory', confidence: 0.85 },
  'Open Palm': { meaning: 'Open hand / Presenting / Stop (context)', confidence: 0.8 },
  Fist: { meaning: 'Disagreement / Emphasis / No', confidence: 0.78 },
  Pointing: { meaning: 'Directing attention', confidence: 0.82 },
};

const MOTION_SIGN_MAP: Record<string, { meaning: string; confidence: number }> = {
  Waving: { meaning: 'Calling for attention / Hello', confidence: 0.92 },
  'Come Here': { meaning: 'Beckoning — come closer', confidence: 0.88 },
  'Stop Sign': { meaning: 'Stop / Wait', confidence: 0.9 },
  Clapping: { meaning: 'Applause / Appreciation', confidence: 0.87 },
};

function pushCue(
  cues: BodyLanguageCue[],
  signal: string,
  meaning: string,
  category: BodyLanguageCue['category'],
  confidence: number,
): void {
  cues.push({ signal, meaning, category, confidence });
}

function nearestDistance(point: { x: number; y: number }, targets: Array<{ x: number; y: number }>): number {
  if (targets.length === 0) return Infinity;
  return Math.min(...targets.map((t) => distance(point, t)));
}

function detectSelfTouch(
  hands: DetectedHand[],
  poseLandmarks: PoseLandmark[],
  faces: DetectedFace[],
): BodyLanguageCue[] {
  const cues: BodyLanguageCue[] = [];
  if (hands.length === 0) return cues;

  const hasPoseFace = poseLandmarks.length >= 11;
  const nose = hasPoseFace ? poseLandmarks[0] : null;
  const leftEye = hasPoseFace ? [poseLandmarks[1], poseLandmarks[2], poseLandmarks[3]] : [];
  const rightEye = hasPoseFace ? [poseLandmarks[4], poseLandmarks[5], poseLandmarks[6]] : [];
  const ears = hasPoseFace ? [poseLandmarks[7], poseLandmarks[8]] : [];
  const mouth = hasPoseFace ? [poseLandmarks[9], poseLandmarks[10]] : [];

  for (const hand of hands) {
    const wrist = hand.landmarks[0];
    const indexTip = hand.landmarks[8];

    for (const face of faces) {
      const center = bboxCenter(face.bbox);
      const faceDist = distance(indexTip, center);
      if (faceDist < 0.12) {
        pushCue(cues, 'Hand on face', 'Touching face — thinking / hiding / discomfort', 'self-touch', 0.84);
      }
    }

    if (nose) {
      const eyeDist = Math.min(
        nearestDistance(indexTip, leftEye),
        nearestDistance(indexTip, rightEye),
        nearestDistance(wrist, leftEye),
        nearestDistance(wrist, rightEye),
      );
      if (eyeDist < 0.07) {
        pushCue(cues, 'Hand near eyes', 'Rubbing eyes — tired / stressed / thinking', 'self-touch', 0.86);
      }

      const chinDist = nearestDistance(indexTip, mouth);
      const nearChin = chinDist < 0.08 && indexTip.y >= nose.y - 0.02;
      if (nearChin) {
        pushCue(
          cues,
          'Hand on chin/jaw',
          'Touching chin or jaw — thinking / scratching beard',
          'self-touch',
          0.83,
        );
      }

      const faceDist = nearestDistance(indexTip, [nose, ...mouth]);
      if (faceDist < 0.09 && !nearChin && eyeDist >= 0.07) {
        pushCue(cues, 'Hand on face', 'Hands on face — self-soothing / thinking', 'self-touch', 0.8);
      }
    }

    if (ears.length) {
      const earDist = nearestDistance(wrist, ears);
      if (earDist < 0.09) {
        pushCue(cues, 'Hand on head', 'Touching head / ear — stress / headache', 'self-touch', 0.82);
      }
    }
  }

  if (hands.length >= 2 && hasPoseFace && nose) {
    const bothNearHead = hands.every((h) => {
      const w = h.landmarks[0];
      return w.y < nose.y + 0.05 && nearestDistance(w, [nose, ...ears]) < 0.12;
    });
    if (bothNearHead) {
      pushCue(cues, 'Hands on head', 'Holding head — overwhelmed / tired / stressed', 'self-touch', 0.88);
    }
  }

  return cues;
}

function detectPostureCues(poseActions: PoseAction[]): BodyLanguageCue[] {
  const cues: BodyLanguageCue[] = [];
  const names = new Set(poseActions.map((a) => a.name));

  if (names.has('Bending') && names.has('Walking')) {
    pushCue(cues, 'Walking bent over', 'Hunched walk — tired / focused downward', 'posture', 0.8);
  } else if (names.has('Bending') && names.has('Standing')) {
    pushCue(cues, 'Leaning forward', 'Leaning — focus / fatigue / interest', 'posture', 0.78);
  } else if (names.has('Bending')) {
    pushCue(cues, 'Bending over', 'Bending — reaching / tired posture', 'posture', 0.75);
  }

  if (names.has('Squat')) {
    pushCue(cues, 'Squatting', 'Low posture — resting / working low', 'posture', 0.76);
  }

  if (names.has('Left Hand Raised') || names.has('Right Hand Raised')) {
    pushCue(cues, 'Hand raised', 'Hand up — question / greeting / attention', 'posture', 0.74);
  }

  return cues;
}

function dedupeCues(cues: BodyLanguageCue[]): BodyLanguageCue[] {
  const best = new Map<string, BodyLanguageCue>();
  for (const cue of cues) {
    const key = `${cue.category}:${cue.meaning}`;
    const existing = best.get(key);
    if (!existing || cue.confidence > existing.confidence) {
      best.set(key, cue);
    }
  }
  return [...best.values()].sort((a, b) => b.confidence - a.confidence);
}

export function interpretBodyLanguage(params: {
  staticGestures: StaticGesture[];
  motionGestures: MotionGesture[];
  poseActions: PoseAction[];
  hands: DetectedHand[];
  poseLandmarks: PoseLandmark[];
  faces: DetectedFace[];
}): BodyLanguageCue[] {
  const cues: BodyLanguageCue[] = [];
  const seenStatic = new Set<string>();

  for (const gesture of params.staticGestures) {
    const mapped = STATIC_SIGN_MAP[gesture.name];
    if (!mapped || seenStatic.has(gesture.name)) continue;
    seenStatic.add(gesture.name);
    pushCue(
      cues,
      gesture.name,
      mapped.meaning,
      'sign',
      Math.min(mapped.confidence, gesture.confidence),
    );
  }

  for (const gesture of params.motionGestures) {
    const mapped = MOTION_SIGN_MAP[gesture.name];
    if (!mapped) continue;
    pushCue(
      cues,
      gesture.name,
      mapped.meaning,
      'motion',
      Math.min(mapped.confidence, gesture.confidence),
    );
  }

  cues.push(...detectPostureCues(params.poseActions));
  cues.push(...detectSelfTouch(params.hands, params.poseLandmarks, params.faces));

  return dedupeCues(cues).slice(0, 8);
}
