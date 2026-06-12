/** L9 — Narrative Builder + internal monologue for Gemma. */

import type { HalLayer, RecentChange, SituationState } from "../types";
import type { AgentState } from "./attentionManager";
import { activityLabelHe } from "./eventFusion";
import type { SceneState } from "./sceneStateEngine";
import type { MetaEvent } from "./eventFusion";

export type NarrativeFrame = {
  whatChanged: string[];
  interpretation: string[];
  contextShift: string;
  suggestedResponseTone: string;
  internalMonologue: string[];
};

type BuildInput = {
  scene: SceneState;
  prevScene: SceneState | null;
  metaEvents: MetaEvent[];
  agent: AgentState;
  recentChanges: RecentChange[];
  situation: SituationState;
  hal: HalLayer;
  personPresent: boolean;
};

const toneFromAgent = (agent: AgentState, hal: HalLayer): string => {
  if (agent.emotionalBias >= 0.35) return "supportive-calm";
  if (agent.curiosity >= 0.55) return "soft-curious";
  if (hal.tone) return String(hal.tone);
  if (agent.verbosity >= 0.5) return "warm-engaged";
  return "quiet-observant";
};

export const buildNarrativeFrame = (input: BuildInput): NarrativeFrame => {
  const { scene, prevScene, metaEvents, agent, recentChanges, situation, hal, personPresent } =
    input;

  const whatChanged: string[] = [];
  for (const c of recentChanges.slice(0, 5)) {
    if (c.ageSec <= 30) whatChanged.push(`${c.text} (${c.ageSec}s ago)`);
  }
  for (const m of metaEvents.slice(0, 3)) {
    whatChanged.push(m.meaning);
  }
  if (!whatChanged.length && personPresent) {
    whatChanged.push(`מצב נוכחי: ${activityLabelHe(scene.activity)}`);
  }

  const interpretation: string[] = metaEvents.map((m) => m.meaning);
  if (situation.description) interpretation.push(situation.description);
  if (hal.interpretation && !interpretation.includes(hal.interpretation)) {
    interpretation.push(hal.interpretation);
  }
  if (hal.sceneLabel) interpretation.push(`סצנה: ${hal.sceneLabel}`);

  let contextShift = "stable — no major shift";
  if (prevScene && prevScene.activity !== scene.activity) {
    contextShift = `transition ${prevScene.activity} → ${scene.activity}`;
  } else if (scene.stability === "chaotic") {
    contextShift = "environment becoming chaotic";
  } else if (scene.stability === "changing") {
    contextShift = "gradual context shift";
  }

  const internalMonologue: string[] = [];
  if (personPresent) {
    internalMonologue.push("I confirm a person is in frame — this is authoritative.");
  } else {
    internalMonologue.push("No person confirmed — stay quiet unless environment shifts.");
  }
  if (metaEvents.some((e) => e.type === "social_interaction_attempt")) {
    internalMonologue.push("I notice the user may be waiting for my attention.");
  }
  if (metaEvents.some((e) => e.type === "waiting_state")) {
    internalMonologue.push("Repeated stillness suggests a waiting state.");
  }
  if (metaEvents.some((e) => e.type === "thinking_state")) {
    internalMonologue.push("Hand-to-face stillness — likely thinking, not interrupting.");
  }
  if (agent.curiosity >= 0.55 && agent.verbosity >= 0.4) {
    internalMonologue.push("Curiosity is high enough to offer a gentle observational line.");
  }
  if (agent.emotionalBias >= 0.35) {
    internalMonologue.push("Stress signals — prioritize calm supportive tone.");
  }

  return {
    whatChanged: whatChanged.slice(0, 6),
    interpretation: interpretation.slice(0, 5),
    contextShift,
    suggestedResponseTone: toneFromAgent(agent, hal),
    internalMonologue: internalMonologue.slice(0, 8),
  };
};

/** Human-readable block for Gemma — narrative-first, not raw sensors. */
export const formatNarrativeForGemma = (params: {
  scene: SceneState;
  metaEvents: MetaEvent[];
  agent: AgentState;
  narrative: NarrativeFrame;
  personPresent: boolean;
  halMood: string;
}): string => {
  const { scene, metaEvents, agent, narrative, personPresent, halMood } = params;
  const lines = [
    "HAL NARRATIVE FRAME (use this — not raw sensor dumps):",
    "",
    "SYSTEM CONTEXT:",
    `- Scene activity: ${scene.activity} (${activityLabelHe(scene.activity)})`,
    `- Stability: ${scene.stability}`,
    `- Focus target: ${scene.focusTarget}`,
    `- Engagement: ${scene.engagementLevel}/100`,
    `- Person in frame: ${personPresent ? "YES (authoritative)" : "NO"}`,
    `- Agent mood: ${halMood}`,
    `- Curiosity: ${(agent.curiosity * 100).toFixed(0)}% | Attentiveness: ${(agent.attentiveness * 100).toFixed(0)}% | Verbosity: ${(agent.verbosity * 100).toFixed(0)}%`,
    "",
    "META EVENTS (fused meaning):",
    ...metaEvents.map(
      (e) => `- [${e.type}] ${e.meaning} (conf ${(e.confidence * 100).toFixed(0)}%, from: ${e.components.join("+")})`,
    ),
    ...(metaEvents.length ? [] : ["- (none significant this tick)"]),
    "",
    "WHAT CHANGED:",
    ...narrative.whatChanged.map((w) => `- ${w}`),
    "",
    "INTERPRETATION:",
    ...narrative.interpretation.map((i) => `- ${i}`),
    "",
    `CONTEXT SHIFT: ${narrative.contextShift}`,
    `SUGGESTED TONE: ${narrative.suggestedResponseTone}`,
    "",
    "INTERNAL MONOLOGUE (reasoning only — do NOT quote to user):",
    ...narrative.internalMonologue.map((m) => `- ${m}`),
    "",
    "TASK: Respond as HAL/Data — aware, contextual, Hebrew when user writes Hebrew. Tentative when uncertain.",
  ];
  return lines.join("\n");
};
