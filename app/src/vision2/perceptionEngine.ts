/** L2 — converts VisionResult to factual observations (no raw landmarks exported). */

import type { VisionResult } from "../vision-lab/core/types";
import type { ObservationSet } from "./types";

const norm = (s: string) => s.trim().toLowerCase();

export const perceiveFromVisionResult = (
  result: VisionResult,
  personPresent: boolean,
  motionLevel = 0,
  now = Date.now(),
): ObservationSet => {
  const bodySignals = new Set(result.bodyLanguage.map((c) => norm(c.signal)));
  const staticNames = new Set(result.staticGestures.map((g) => norm(g.name)));
  const motionNames = new Set(result.motionGestures.map((g) => norm(g.name)));
  const interactionNames = new Set(result.interactions.map((i) => norm(i.name)));

  const face = result.faces[0];
  const gaze = face?.gazeDirection?.toLowerCase() ?? "";

  return {
    timestamp: now,
    personPresent,
    touchingFace: bodySignals.has("hand on face"),
    touchingHead: bodySignals.has("hand on head"),
    handsOnHead: bodySignals.has("hands on head"),
    handNearEyes: bodySignals.has("hand near eyes"),
    handOnChin: bodySignals.has("hand on chin/jaw"),
    raisedHand:
      result.poseActions.some((a) => /hand raised/i.test(a.name)) ||
      bodySignals.has("hand raised"),
    waving: motionNames.has("waving"),
    pointing: staticNames.has("pointing"),
    thumbsUp: staticNames.has("thumbs up"),
    thumbsDown: staticNames.has("thumbs down"),
    holdingCup: interactionNames.has("holding cup"),
    usingPhone: interactionNames.has("using phone"),
    gazeDown: /down/i.test(gaze),
    gazeAtCamera: /center|camera/i.test(gaze),
    motionLevel: Math.max(0, Math.min(1, motionLevel)),
  };
};
