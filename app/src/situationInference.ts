/** Situation inference — held objects, posture + drink combos (browser heuristics). */

import type { MotionSnapshot } from "./motionLayer";
import type { BBox, Keypoint, PersonPoseState } from "./poseHeuristics";
import { analyzePersonPose, attachHoldingObjects } from "./poseHeuristics";
import { makeSemanticEvent, type SemanticEvent } from "./worldMemory";

export const HOLDABLE_OBJECTS = new Set([
  "cup",
  "bottle",
  "book",
  "phone",
  "keyboard",
  "bag",
  "backpack",
]);

export const SITUATION_DEBOUNCE = {
  urgentMs: 12_000,
  normalMs: 45_000,
} as const;

const isDrink = (label: string): boolean => /cup|bottle/.test(label);
const isHoldable = (label: string): boolean => HOLDABLE_OBJECTS.has(label);

export type SituationInput = {
  keypoints: Keypoint[] | null;
  personBbox: BBox | null;
  objectBoxes: { label: string; bbox: BBox }[];
  motion: MotionSnapshot;
  personInFrame: boolean;
};

export type SituationContext = {
  prevPose: PersonPoseState | null;
  prevHolding: string[];
};

export const detectSituationEvents = (
  input: SituationInput,
  ctx: SituationContext,
  canEmit: (kind: string, urgent: boolean) => boolean,
): { pose: PersonPoseState; events: SemanticEvent[]; holding: string[] } => {
  const holding = input.personBbox
    ? attachHoldingObjects(input.personBbox, input.objectBoxes, 0.2, input.keypoints)
    : [];
  const pose = analyzePersonPose(input.keypoints, holding);
  const events: SemanticEvent[] = [];

  if (!input.personInFrame) {
    return { pose, events, holding };
  }

  const newHeld = holding.filter((h) => !ctx.prevHolding.includes(h) && isHoldable(h));
  for (const item of newHeld) {
    const kind = `held_${item}`;
    if (canEmit(kind, true)) {
      events.push(
        makeSemanticEvent(
          "activity_change",
          `Person now holding ${item}`,
          `object_held:${item}`,
          true,
        ),
      );
    }
  }

  const stoodUp =
    ctx.prevPose?.poseState === "sitting" && pose.poseState === "standing";

  if (stoodUp && canEmit("pose_change", false)) {
    /* pose_change emitted by SituationEngine tracker — avoid duplicate */
  }

  const drinkHeld = holding.find(isDrink);
  if (stoodUp && drinkHeld && canEmit("stood_with_drink", true)) {
    events.push(
      makeSemanticEvent(
        "activity_change",
        `Stood up while holding ${drinkHeld}`,
        "stood_with_drink",
        true,
      ),
    );
  } else if (
    !stoodUp &&
    pose.gestures.includes("wave") &&
    canEmit("pose_wave", true) &&
    !input.motion.waveLike
  ) {
    events.push(
      makeSemanticEvent("activity_change", "Pose: raised hand / wave gesture", "wave", true),
    );
  } else if (pose.gestures.includes("focused_work") && canEmit("focused_work", false)) {
    events.push(
      makeSemanticEvent(
        "activity_change",
        "Person appears focused on work surface",
        "focused_work",
        true,
      ),
    );
  }

  return { pose, events, holding };
};
