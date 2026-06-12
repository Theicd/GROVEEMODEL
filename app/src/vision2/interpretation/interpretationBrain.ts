/** L9 — Interpretation Brain orchestrator: state → meaning → narrative. */

import type { ObservationSet } from "../types";
import type { DialogueContext, HalLayer } from "../types";
import { createAgentState, resetAgentState, updateAgentState, type AgentState } from "./attentionManager";
import { fuseMetaEvents, type MetaEvent } from "./eventFusion";
import { buildNarrativeFrame, formatNarrativeForGemma, type NarrativeFrame } from "./narrativeBuilder";
import { computeSceneState, type SceneActivity, type SceneState } from "./sceneStateEngine";

export type InterpretationLayer = {
  sceneState: SceneState;
  metaEvents: MetaEvent[];
  agentState: AgentState;
  narrative: NarrativeFrame;
  gemmaBlock: string;
};

export class InterpretationBrain {
  private agent = createAgentState();
  private prevScene: SceneState | null = null;
  private prevActivity: SceneActivity | null = null;
  private prevMotion = 0;

  reset(): void {
    this.agent = createAgentState();
    this.prevScene = null;
    this.prevActivity = null;
    this.prevMotion = 0;
  }

  process(input: {
    dialogue: DialogueContext;
    obs: ObservationSet;
    hal: HalLayer;
    faceTouchSec: number;
    waveRising: boolean;
    personJustEntered: boolean;
    msSinceUserChat: number;
    now?: number;
  }): InterpretationLayer {
    const now = input.now ?? Date.now();
    const { dialogue, obs, hal } = input;
    const motionDelta = Math.abs(obs.motionLevel - this.prevMotion);
    this.prevMotion = obs.motionLevel;

    const sceneState = computeSceneState({
      obs,
      human: dialogue.personState,
      body: dialogue.bodyLanguage,
      situation: dialogue.situation,
      recentChanges: dialogue.recentChanges,
      prevActivity: this.prevActivity,
      motionDelta,
      now,
    });

    const metaEvents = fuseMetaEvents({
      obs,
      human: dialogue.personState,
      body: dialogue.bodyLanguage,
      scene: sceneState,
      recentChanges: dialogue.recentChanges,
      faceTouchSec: input.faceTouchSec,
      waveRising: input.waveRising,
      personJustEntered: input.personJustEntered,
    });

    this.agent = updateAgentState(this.agent, {
      scene: sceneState,
      metaEvents,
      personPresent: hal.personPresent,
      stressLevel: hal.stressLevel,
      msSinceUserChat: input.msSinceUserChat,
      now,
    });

    const narrative = buildNarrativeFrame({
      scene: sceneState,
      prevScene: this.prevScene,
      metaEvents,
      agent: this.agent,
      recentChanges: dialogue.recentChanges,
      situation: dialogue.situation,
      hal,
      personPresent: hal.personPresent,
    });

    const gemmaBlock = formatNarrativeForGemma({
      scene: sceneState,
      metaEvents,
      agent: this.agent,
      narrative,
      personPresent: hal.personPresent,
      halMood: `${hal.mood} / ${hal.moodLabelHe}`,
    });

    this.prevScene = sceneState;
    this.prevActivity = sceneState.activity;

    return {
      sceneState,
      metaEvents,
      agentState: this.agent,
      narrative,
      gemmaBlock,
    };
  }
}

export { resetAgentState };
