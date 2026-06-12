/** L9 — Event Fusion: components → meta-meaning. */

import type { ObservationSet } from "../types";
import type { BodyLanguageVector, HumanState, RecentChange } from "../types";
import type { SceneActivity, SceneState } from "./sceneStateEngine";

export type MetaEvent = {
  type: string;
  confidence: number;
  components: string[];
  meaning: string;
  ageSec: number;
};

type FusionInput = {
  obs: ObservationSet;
  human: HumanState;
  body: BodyLanguageVector;
  scene: SceneState;
  recentChanges: RecentChange[];
  faceTouchSec: number;
  waveRising: boolean;
  personJustEntered: boolean;
};

const push = (
  out: MetaEvent[],
  type: string,
  confidence: number,
  components: string[],
  meaning: string,
  ageSec = 0,
): void => {
  if (out.some((e) => e.type === type && e.ageSec === ageSec)) return;
  out.push({ type, confidence, components, meaning, ageSec });
};

export const fuseMetaEvents = (input: FusionInput): MetaEvent[] => {
  const { obs, body, scene, recentChanges, faceTouchSec, waveRising, personJustEntered } = input;
  const out: MetaEvent[] = [];

  if (obs.waving && obs.personPresent) {
    push(
      out,
      "social_interaction_attempt",
      scene.focusTarget === "agent" ? 0.88 : 0.72,
      ["waving", "person_present"],
      "User tries to engage attention without speech",
    );
  }

  if (personJustEntered || recentChanges.some((c) => c.kind === "entered" && c.ageSec <= 5)) {
    push(
      out,
      "presence_established",
      0.9,
      ["person_entered"],
      "New human presence in scene",
      recentChanges.find((c) => c.kind === "entered")?.ageSec ?? 0,
    );
  }

  if (waveRising && personJustEntered) {
    push(
      out,
      "greeting_burst",
      0.85,
      ["person_entered", "waving"],
      "Social greeting sequence — interaction opening",
    );
  }

  if ((obs.touchingFace || obs.handOnChin) && obs.motionLevel < 0.15 && faceTouchSec >= 3) {
    push(
      out,
      "thinking_state",
      Math.min(0.95, 0.5 + body.thinking),
      ["hand_on_face", "stillness"],
      "User appears to be processing or deciding",
    );
  }

  if (obs.usingPhone && obs.personPresent) {
    push(
      out,
      "attention_diverted",
      0.78,
      ["phone_usage", "person_present"],
      "Attention split toward personal device",
    );
  }

  if (body.stressed >= 0.55 && (obs.handsOnHead || obs.touchingHead)) {
    push(
      out,
      "overload_signal",
      body.stressed,
      ["hands_on_head", "stress_score"],
      "Cognitive or emotional overload may be building",
    );
  }

  if (scene.stability === "chaotic" && scene.activity === "movement") {
    push(
      out,
      "high_motion_burst",
      scene.intensity,
      ["motion", "instability"],
      "Elevated physical activity — environment unstable",
    );
  }

  if (scene.activity === "focused_work" && body.focused >= 0.6) {
    push(
      out,
      "deep_work_block",
      body.focused,
      ["focused_work", "low_motion"],
      "Sustained task focus — interrupt only if necessary",
    );
  }

  if (scene.activity === "idle" && obs.personPresent && scene.engagementLevel < 25) {
    push(
      out,
      "waiting_state",
      0.65,
      ["stillness", "low_engagement"],
      "User may be waiting — curiosity appropriate",
    );
  }

  if (obs.pointing && scene.focusTarget === "agent") {
    push(
      out,
      "direct_reference",
      0.8,
      ["pointing", "agent_focus"],
      "User references something or seeks confirmation",
    );
  }

  return out.sort((a, b) => b.confidence - a.confidence).slice(0, 6);
};

export const activityLabelHe = (a: SceneActivity): string => {
  const map: Record<SceneActivity, string> = {
    idle: "ממתין",
    interaction: "אינטראקציה",
    attention: "תשומת לב",
    confusion: "עיבוד/הסתייגות",
    social: "חברתי",
    movement: "תנועה",
    focused_work: "עבודה ממוקדת",
  };
  return map[a] ?? a;
};
