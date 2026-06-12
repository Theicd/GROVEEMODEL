/** L8 — unified world snapshot from world memory + vision2 layers. */

import type { WorldMemory } from "../worldMemory";
import type {
  BodyLanguageVector,
  EnvironmentKind,
  HumanState,
  PersonModel,
  RoomModel,
  SessionModel,
  SituationPrimary,
  SituationState,
  WorldSnapshot,
} from "./types";

const mapEnvironment = (env: string, objects: string[]): EnvironmentKind => {
  const e = env.toLowerCase();
  if (/office|desk/.test(e)) return "office";
  if (/kitchen/.test(e)) return "kitchen";
  if (/living|sofa/.test(e)) return "living";
  if (/bedroom|bed/.test(e)) return "bedroom";
  if (objects.some((o) => /laptop|keyboard|monitor/.test(o))) return "office";
  if (objects.some((o) => /cup|bottle|refrigerator|oven/.test(o))) return "kitchen";
  if (objects.some((o) => /couch|sofa|tv/.test(o))) return "living";
  if (objects.some((o) => /bed/.test(o))) return "bedroom";
  return "unknown";
};

export const buildRoomModel = (world: WorldMemory): RoomModel => {
  const objs = world.objects;
  return {
    hasLaptop: objs.some((o) => /laptop|computer|keyboard/.test(o)),
    hasCup: objs.some((o) => /cup|bottle|coffee/.test(o)),
    hasPhone: objs.some((o) => /phone|cell/.test(o)),
    hasTv: objs.some((o) => /tv|television|monitor/.test(o)),
    environment: mapEnvironment(world.environment, objs),
    stableObjects: [...world.baselineObjects.slice(0, 8)],
    semanticNotes: world.bootContext.trim().slice(0, 280),
  };
};

export type SessionTracker = {
  workStartedAt: number;
  lastBreakAt: number;
  lastGreetingAt: number;
};

export const createSessionTracker = (): SessionTracker => ({
  workStartedAt: 0,
  lastBreakAt: 0,
  lastGreetingAt: 0,
});

export const buildSessionModel = (
  world: WorldMemory,
  session: SessionTracker,
  faceTouchSec: number,
  _situationPrimary: SituationPrimary,
  now = Date.now(),
): SessionModel => {
  const sceneAgeSec = world.sceneAgeSec;
  let workSessionMin = 0;
  if (session.workStartedAt > 0) {
    workSessionMin = Math.floor((now - session.workStartedAt) / 60_000);
  }
  return {
    sceneAgeSec,
    workSessionMin,
    lastBreakAt: session.lastBreakAt,
    lastGreetingAt: session.lastGreetingAt,
    faceTouchDurationSec: Math.round(faceTouchSec),
  };
};

export const buildPersonModel = (
  world: WorldMemory,
  human: HumanState,
  situation: SituationState,
): PersonModel => ({
  present: world.personPresent,
  absentDurationSec: world.personPresent ? 0 : Math.floor(world.msSinceAbsent() / 1000),
  posture: human.posture,
  attention: human.attention,
  activity: human.activity,
  reflecting: situation.primary === "reflecting",
});

export const buildWorldSnapshot = (
  world: WorldMemory,
  human: HumanState,
  body: BodyLanguageVector,
  situation: SituationState,
  session: SessionTracker,
  faceTouchSec: number,
  now = Date.now(),
): WorldSnapshot => ({
  room: buildRoomModel(world),
  person: buildPersonModel(world, human, situation),
  session: buildSessionModel(world, session, faceTouchSec, situation.primary, now),
  bodyLanguage: body,
  situation,
  updatedAt: now,
});

export const updateSessionTracker = (
  tracker: SessionTracker,
  situation: SituationState,
  obs: { waving: boolean; holdingCup: boolean; personPresent: boolean },
  now = Date.now(),
): void => {
  if (obs.waving && obs.personPresent) {
    tracker.lastGreetingAt = now;
  }
  if (situation.primary === "working" && obs.personPresent) {
    if (!tracker.workStartedAt) tracker.workStartedAt = now;
  } else if (situation.primary === "drinking" || situation.primary === "idle") {
    if (tracker.workStartedAt) {
      tracker.lastBreakAt = now;
      tracker.workStartedAt = 0;
    }
  }
  if (!obs.personPresent) {
    tracker.workStartedAt = 0;
  }
};
