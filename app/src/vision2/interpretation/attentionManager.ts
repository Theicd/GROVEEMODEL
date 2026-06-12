/** L9 — Attention Manager: HAL/Data agent drives curiosity + verbosity. */

import type { SceneState } from "./sceneStateEngine";
import type { MetaEvent } from "./eventFusion";

export type AgentState = {
  curiosity: number;
  attentiveness: number;
  verbosity: number;
  emotionalBias: number;
  lastMeaningfulChangeAt: number;
  updatedAt: number;
};

export const createAgentState = (): AgentState => ({
  curiosity: 0.35,
  attentiveness: 0.4,
  verbosity: 0.35,
  emotionalBias: 0,
  lastMeaningfulChangeAt: Date.now(),
  updatedAt: Date.now(),
});

type UpdateInput = {
  scene: SceneState;
  metaEvents: MetaEvent[];
  personPresent: boolean;
  stressLevel: number;
  msSinceUserChat: number;
  now?: number;
};

const clamp = (n: number) => Math.max(0, Math.min(1, n));

export const updateAgentState = (state: AgentState, input: UpdateInput): AgentState => {
  const now = input.now ?? Date.now();
  const { scene, metaEvents, personPresent, stressLevel, msSinceUserChat } = input;

  let curiosity = state.curiosity;
  let attentiveness = state.attentiveness;
  let verbosity = state.verbosity;
  let emotionalBias = state.emotionalBias;
  let lastMeaningfulChangeAt = state.lastMeaningfulChangeAt;

  const meaningful = metaEvents.some((e) => e.confidence >= 0.7) || scene.stability !== "stable";
  if (meaningful) {
    lastMeaningfulChangeAt = now;
    curiosity = clamp(curiosity + 0.08);
    attentiveness = clamp(attentiveness + 0.12);
  } else {
    const idleSec = (now - lastMeaningfulChangeAt) / 1000;
    if (idleSec > 12 && personPresent) curiosity = clamp(curiosity + 0.04);
    if (idleSec > 45) verbosity = clamp(verbosity + 0.03);
  }

  if (scene.activity === "movement" || scene.stability === "chaotic") {
    attentiveness = clamp(attentiveness + 0.1);
  }

  if (scene.activity === "social" || scene.focusTarget === "agent") {
    verbosity = clamp(verbosity + 0.15);
    curiosity = clamp(curiosity + 0.1);
  }

  if (!personPresent) {
    curiosity = clamp(curiosity - 0.06);
    attentiveness = clamp(attentiveness - 0.08);
    verbosity = clamp(verbosity - 0.1);
  }

  if (stressLevel >= 0.5) {
    emotionalBias = clamp(stressLevel * 0.6 - 0.1);
    verbosity = clamp(verbosity - 0.05);
  } else {
    emotionalBias = clamp(emotionalBias - 0.02);
  }

  if (msSinceUserChat < 30_000) {
    verbosity = clamp(verbosity - 0.12);
  }

  return {
    curiosity: clamp(curiosity * 0.98 + 0.01),
    attentiveness: clamp(attentiveness * 0.98 + 0.02),
    verbosity: clamp(verbosity * 0.97 + 0.015),
    emotionalBias: clamp(emotionalBias),
    lastMeaningfulChangeAt,
    updatedAt: now,
  };
};

export const resetAgentState = (): AgentState => createAgentState();
