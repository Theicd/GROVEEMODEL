/** Default interpretation layer before InterpretationBrain runs. */

import type { InterpretationLayer } from "./types";

export const EMPTY_INTERPRETATION: InterpretationLayer = {
  sceneState: {
    activity: "idle",
    intensity: 0,
    stability: "stable",
    focusTarget: "none",
    engagementLevel: 0,
  },
  metaEvents: [],
  agentState: {
    curiosity: 0.35,
    attentiveness: 0.4,
    verbosity: 0.35,
    emotionalBias: 0,
  },
  narrative: {
    whatChanged: [],
    interpretation: [],
    contextShift: "initializing",
    suggestedResponseTone: "quiet-observant",
  },
  gemmaBlock: "HAL narrative: initializing perception…",
};

export const toInterpretationLayer = (layer: import("./interpretation/interpretationBrain").InterpretationLayer): InterpretationLayer => ({
  sceneState: {
    activity: layer.sceneState.activity,
    intensity: layer.sceneState.intensity,
    stability: layer.sceneState.stability,
    focusTarget: layer.sceneState.focusTarget,
    engagementLevel: layer.sceneState.engagementLevel,
  },
  metaEvents: layer.metaEvents.map((e) => ({
    type: e.type,
    confidence: e.confidence,
    components: e.components,
    meaning: e.meaning,
  })),
  agentState: {
    curiosity: layer.agentState.curiosity,
    attentiveness: layer.agentState.attentiveness,
    verbosity: layer.agentState.verbosity,
    emotionalBias: layer.agentState.emotionalBias,
  },
  narrative: {
    whatChanged: layer.narrative.whatChanged,
    interpretation: layer.narrative.interpretation,
    contextShift: layer.narrative.contextShift,
    suggestedResponseTone: layer.narrative.suggestedResponseTone,
  },
  gemmaBlock: layer.gemmaBlock,
});
