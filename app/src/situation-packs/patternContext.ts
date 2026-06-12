/** Build normalized match context from Vision 2.0 layers + lab result. */

import type { VisionResult } from "../vision-lab/core/types";
import type { SemanticEvent } from "../worldMemory";
import type { WorldMemory } from "../worldMemory";
import type {
  BodyLanguageVector,
  CapabilityContext,
  HumanState,
  ObservationSet,
  SituationState,
  WorldSnapshot,
} from "../vision2/types";
import type { SignalHistory } from "./signalHistory";
import { silenceSec } from "./signalHistory";

const norm = (s: string) => s.trim().toLowerCase().replace(/\s+/g, "_");

export type MatchContext = {
  now: number;
  obs: ObservationSet;
  human: HumanState;
  body: BodyLanguageVector;
  situation: SituationState;
  world: WorldMemory;
  snapshot: WorldSnapshot;
  capabilities?: CapabilityContext;
  history: SignalHistory;
  gestures: string[];
  bodyLanguage: string[];
  events: string[];
  objects: string[];
  poseActions: string[];
  motionLevel: number;
  silenceSec: number;
  personJustEntered: boolean;
};

export const collectSignalsFromResult = (result: VisionResult): {
  gestures: string[];
  bodyLanguage: string[];
  events: string[];
  objects: string[];
  poseActions: string[];
} => {
  const gestures = [
    ...result.staticGestures.map((g) => norm(g.name)),
    ...result.motionGestures.map((g) => norm(g.name)),
  ];
  const bodyLanguage = result.bodyLanguage.map((c) => norm(c.signal));
  const events = result.events.map((e) => norm(e.name));
  const objects = result.objects.map((o) => norm(o.displayLabel || o.label));
  const poseActions = result.poseActions.map((a) => norm(a.name));
  return { gestures, bodyLanguage, events, objects, poseActions };
};

export const bodyLanguageFromObs = (obs: ObservationSet): string[] => {
  const out: string[] = [];
  if (obs.touchingFace) out.push("hand_on_face");
  if (obs.handOnChin) out.push("hand_on_chin");
  if (obs.handsOnHead || obs.touchingHead) out.push("hands_on_head");
  if (obs.handNearEyes) out.push("hand_near_eyes");
  return out;
};

export const gesturesFromObs = (obs: ObservationSet): string[] => {
  const out: string[] = [];
  if (obs.waving) out.push("waving");
  if (obs.pointing) out.push("pointing");
  if (obs.thumbsUp) out.push("thumbs_up");
  if (obs.thumbsDown) out.push("thumbs_down");
  if (obs.raisedHand) out.push("hand_raised");
  return out;
};

export const recordObservationsToHistory = (
  history: SignalHistory,
  obs: ObservationSet,
  signals: ReturnType<typeof collectSignalsFromResult>,
  now: number,
): void => {
  for (const g of [...gesturesFromObs(obs), ...signals.gestures]) {
    if (g) recordKey(history, `gesture:${g}`, now);
  }
  for (const b of [...bodyLanguageFromObs(obs), ...signals.bodyLanguage]) {
    if (b) recordKey(history, `body:${b}`, now);
  }
  for (const e of signals.events) recordKey(history, `event:${e}`, now);
  for (const o of signals.objects) recordKey(history, `object:${o}`, now);
  if (obs.holdingCup) recordKey(history, "object:cup", now);
  if (obs.usingPhone) recordKey(history, "event:phone_usage", now);
  if (obs.personPresent) recordKey(history, "presence:person", now);
};

const recordKey = (history: SignalHistory, key: string, now: number): void => {
  const prev = history.hits.get(key) ?? [];
  history.hits.set(key, [...prev.slice(-40), now]);
  history.lastInteractionAt = now;
};

export const buildMatchContext = (
  result: VisionResult,
  obs: ObservationSet,
  human: HumanState,
  body: BodyLanguageVector,
  situation: SituationState,
  world: WorldMemory,
  snapshot: WorldSnapshot,
  history: SignalHistory,
  freshEvents: SemanticEvent[],
  options?: { capabilities?: CapabilityContext; personJustEntered?: boolean },
): MatchContext => {
  const now = Date.now();
  const collected = collectSignalsFromResult(result);
  const gestureSet = new Set([...gesturesFromObs(obs), ...collected.gestures]);
  const bodySet = new Set([...bodyLanguageFromObs(obs), ...collected.bodyLanguage]);
  const eventSet = new Set([
    ...collected.events,
    ...freshEvents.map((e) => norm(e.text ?? e.subject ?? "")),
  ]);
  if (options?.personJustEntered) {
    eventSet.add("person_entered");
  }

  return {
    now,
    obs,
    human,
    body,
    situation,
    world,
    snapshot,
    capabilities: options?.capabilities,
    history,
    gestures: [...gestureSet],
    bodyLanguage: [...bodySet],
    events: [...eventSet],
    objects: collected.objects,
    poseActions: collected.poseActions,
    motionLevel: obs.motionLevel,
    silenceSec: silenceSec(history, now),
    personJustEntered: options?.personJustEntered ?? false,
  };
};
