/**
 * Sync VisionPipeline results into GROVEE WorldMemory + SemanticEvents.
 */

import type { VisionResult } from "./vision-lab/core/types";
import {
  mapLabGestures,
  mapLabHolding,
  mapPoseActionToState,
} from "./visionBridge";
import { loadSituationRegistry } from "./situationRegistry";
import {
  createSituationTriggerState,
  evaluateSituationTriggers,
  registrySubjectFromLabEvent,
  type SituationTriggerState,
} from "./situationTriggerEngine";
import {
  makeSemanticEvent,
  normalizeLabel,
  type SemanticEvent,
  type WorldMemory,
  type WorldUpdateResult,
} from "./worldMemory";

const PERSON_CONFIRM_FRAMES = 2;

const labEventToSemantic = (name: string, confidence: number): SemanticEvent | null => {
  const n = name.toLowerCase();
  if (/calling for attention|greeting/i.test(n)) {
    return makeSemanticEvent("activity_change", name, "wave", true);
  }
  if (/drinking|holding cup/i.test(n)) {
    return makeSemanticEvent("activity_change", name, "stood_with_drink", true);
  }
  if (/phone usage/i.test(n)) {
    return makeSemanticEvent("activity_change", name, "focused_work", true);
  }
  if (/applause|clapping/i.test(n)) {
    return makeSemanticEvent("activity_change", name, "arm_movement", true);
  }
  if (/like|thumbs up/i.test(n)) {
    return makeSemanticEvent("activity_change", name, "gesture:thumbs_up", confidence >= 0.85);
  }
  if (/jumping|running/i.test(n)) {
    return makeSemanticEvent("activity_change", name, "motion_burst", true);
  }
  return makeSemanticEvent("activity_change", name, `lab:${normalizeLabel(name)}`, false);
};

export type GroveeVisionSyncState = {
  personStreak: number;
  lastEventKeys: Set<string>;
  situationTriggers: SituationTriggerState;
};

export const createGroveeVisionSyncState = (): GroveeVisionSyncState => ({
  personStreak: 0,
  lastEventKeys: new Set(),
  situationTriggers: createSituationTriggerState(),
});

export type GroveeVisionSyncResult = {
  worldUpdate: WorldUpdateResult;
  labEvents: SemanticEvent[];
  personPresent: boolean;
  personJustConfirmed: boolean;
  personJustLeft: boolean;
};

export const syncVisionResultToWorld = (
  world: WorldMemory,
  result: VisionResult,
  syncState: GroveeVisionSyncState,
): GroveeVisionSyncResult => {
  const objectLabels = result.objects.map((o) => normalizeLabel(o.displayLabel || o.label));
  const rawPersonCount = result.objects.filter((o) => o.label === "person" && o.confidence >= 0.45).length;
  const rawHasPerson = rawPersonCount > 0;

  if (rawHasPerson) syncState.personStreak = Math.min(PERSON_CONFIRM_FRAMES, syncState.personStreak + 1);
  else syncState.personStreak = 0;

  const debouncedHasPerson = syncState.personStreak >= PERSON_CONFIRM_FRAMES;
  const prevHadPerson = world.personPresent;
  const people = debouncedHasPerson ? ["person"] : [];

  const worldUpdate = world.applyLightDetection({
    objects: objectLabels.filter((l) => l !== "person"),
    people,
  });

  if (!debouncedHasPerson) {
    world.clearPersonLayer();
    world.pruneStalePersonActivity();
  } else {
    const pose = mapPoseActionToState(result.poseActions);
    world.poseState = pose.poseState;
    world.poseConfidence = pose.confidence;
    world.poseUpdatedAt = Date.now();
    world.poseSource = "vision-lab";
    world.gestures = mapLabGestures(result);
    world.holding = mapLabHolding(result);
    world.focusHint = result.interactions.length
      ? result.interactions.map((i) => i.name).join(", ")
      : "";
  }

  world.richSceneDescription = result.sceneDescription?.trim() ?? "";
  world.environment = result.environment ?? "Unknown";
  world.labBodyLanguage = result.bodyLanguage
    .slice(0, 4)
    .map((c) => `${c.signal}: ${c.meaning}`);
  world.emotionEstimate = result.emotion?.dominant
    ? `${result.emotion.dominant} (~${Math.round(result.emotion.dominantScore * 100)}%)`
    : "";

  world.fingerStates = result.fingerStates.map((f) => ({
    hand: f.hand,
    count: f.count,
    thumb: f.fingers.thumb,
    index: f.fingers.index,
    middle: f.fingers.middle,
    ring: f.fingers.ring,
    pinky: f.fingers.pinky,
  }));

  if (result.faces.length) {
    const f = result.faces[0];
    world.faceSummary = `age~${Math.round(f.estimatedAge)}, gender est. ${f.estimatedGender}, gaze ${f.gazeDirection}`;
  } else {
    world.faceSummary = "";
  }

  world.lastVisionFrameAt = Date.now();

  if (worldUpdate.isBaselineCapture && result.sceneDescription?.trim()) {
    world.lastSummary = result.sceneDescription.trim().slice(0, 320);
    world.baselineEstablished = true;
  } else if (!world.lastSummary.trim() && result.sceneDescription?.trim()) {
    world.lastSummary = result.sceneDescription.trim().slice(0, 320);
  }

  const situationRules = loadSituationRegistry();
  const labEvents: SemanticEvent[] = [];
  for (const ev of result.events) {
    const key = `${ev.name}:${Math.round(ev.confidence * 100)}`;
    if (syncState.lastEventKeys.has(key)) continue;
    syncState.lastEventKeys.add(key);
    if (syncState.lastEventKeys.size > 24) {
      const first = syncState.lastEventKeys.values().next().value;
      if (first) syncState.lastEventKeys.delete(first);
    }
    const subject =
      registrySubjectFromLabEvent(ev.name, situationRules) ??
      labEventToSemantic(ev.name, ev.confidence)?.subject;
    const sem = subject
      ? makeSemanticEvent("activity_change", ev.name, subject, true)
      : labEventToSemantic(ev.name, ev.confidence);
    if (sem) {
      world.lastSituationSubject = sem.subject ?? "";
      world.lastSituationAt = Date.now();
      labEvents.push(sem);
    }
  }

  const registryEvents = evaluateSituationTriggers(
    result,
    situationRules,
    syncState.situationTriggers,
  );
  for (const ev of registryEvents) {
    if (!labEvents.some((e) => e.subject === ev.subject)) {
      world.lastSituationSubject = ev.subject ?? "";
      world.lastSituationAt = Date.now();
      labEvents.push(ev);
    }
  }

  if (labEvents.length) {
    world.applySemanticEvents(labEvents);
  }

  return {
    worldUpdate,
    labEvents,
    personPresent: debouncedHasPerson,
    personJustConfirmed: !prevHadPerson && debouncedHasPerson,
    personJustLeft: prevHadPerson && !debouncedHasPerson,
  };
};
