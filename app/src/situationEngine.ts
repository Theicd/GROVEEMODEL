/** Fuse pose + motion + holding into semantic events for CharacterBrain. */

import { inferPoseStateWithConfidence, type PoseInference, type PersonPoseState } from "./poseHeuristics";
import { PoseStateTracker } from "./poseStateTracker";
import {
  detectSituationEvents,
  SITUATION_DEBOUNCE,
  type SituationInput,
} from "./situationInference";
import { makeSemanticEvent, type SemanticEvent, type WorldMemory } from "./worldMemory";

export type { SituationInput };

export class SituationEngine {
  private lastEventAt = new Map<string, number>();
  private prevPoseState: PersonPoseState | null = null;
  private prevHolding: string[] = [];
  private readonly poseTracker = new PoseStateTracker();
  private lastRawInference: PoseInference = { poseState: "unknown", confidence: 0 };

  reset(): void {
    this.lastEventAt.clear();
    this.prevPoseState = null;
    this.prevHolding = [];
    this.poseTracker.reset();
    this.lastRawInference = { poseState: "unknown", confidence: 0 };
  }

  getLastRawInference() {
    return this.lastRawInference;
  }

  private canEmit = (kind: string, urgent: boolean): boolean => {
    const last = this.lastEventAt.get(kind) ?? 0;
    const windowMs = urgent ? SITUATION_DEBOUNCE.urgentMs : SITUATION_DEBOUNCE.normalMs;
    if (Date.now() - last < windowMs) return false;
    this.lastEventAt.set(kind, Date.now());
    return true;
  };

  analyze(input: SituationInput): { pose: PersonPoseState; events: SemanticEvent[] } {
    const raw = input.keypoints?.length
      ? inferPoseStateWithConfidence(input.keypoints)
      : { poseState: "unknown" as const, confidence: 0 };
    this.lastRawInference = raw;

    const track = this.poseTracker.observe(raw);
    const { pose, events, holding } = detectSituationEvents(
      input,
      { prevPose: this.prevPoseState, prevHolding: this.prevHolding },
      this.canEmit,
    );

    if (this.poseTracker.committedState !== "unknown") {
      pose.poseState = this.poseTracker.committedState;
      pose.confidence = this.poseTracker.committedConfidence;
    } else if (raw.confidence >= 0.45) {
      pose.poseState = raw.poseState;
      pose.confidence = raw.confidence;
    } else {
      pose.poseState = "unknown";
      pose.confidence = raw.confidence;
    }

    if (track.changed && this.canEmit("pose_change", false)) {
      events.unshift(
        makeSemanticEvent(
          "activity_change",
          `Pose changed from ${track.from} to ${track.to}`,
          `pose_change:${track.from}_to_${track.to}`,
          true,
        ),
      );
    }

    this.prevPoseState = pose;
    this.prevHolding = holding;
    return { pose, events };
  }

  applyToWorld(
    world: WorldMemory,
    pose: PersonPoseState,
    events: SemanticEvent[],
    poseSource?: string,
  ): void {
    world.poseState = pose.poseState;
    world.poseConfidence = pose.confidence;
    world.poseUpdatedAt = Date.now();
    world.gestures = pose.gestures;
    world.holding = pose.holding;
    world.focusHint = pose.focusHint;
    if (poseSource) world.poseSource = poseSource;
    if (events.length) world.recordPublicEvents(events);
  }
}

export type { PersonPoseState };
