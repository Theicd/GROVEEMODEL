/**
 * Pattern-based situation packs — HAL/Data style.
 * Situation = bundle of signals + duration + context, not single gesture → reply.
 */

import type { AttentionKind, PostureKind, SituationPrimary } from "../vision2/types";

export type SituationTone =
  | "warm"
  | "soft"
  | "calm"
  | "positive"
  | "curious"
  | "neutral"
  | "friendly"
  | "quiet"
  | "supportive"
  | "playful"
  | "observant"
  | "engaged"
  | "analytical"
  | "relaxed";

export type SituationPriority = "critical" | "high" | "medium" | "low";

export type PackTriggers = {
  /** Any of these gestures (normalized: waving, thumbs_up, pointing…) */
  gestures?: string[];
  bodyLanguage?: string[];
  posture?: PostureKind[];
  attention?: AttentionKind[];
  /** Semantic event types or subjects */
  events?: string[];
  objects?: string[];
  poseActions?: string[];
  /** Vision2 situation.primary */
  situations?: SituationPrimary[];
  /** Signal must hold this many seconds (uses signal history) */
  minDurationSec?: number;
  /** Count in timeWindowSec */
  minRepetition?: number;
  timeWindowSec?: number;
  motion?: "low" | "high" | "variable";
  hands?: "inactive" | "active";
  minBodyScore?: Partial<{ focused: number; thinking: number; stressed: number; bored: number }>;
  minEngagement?: number;
  maxEngagement?: number;
  /** No meaningful motion / interaction */
  minSilenceSec?: number;
  personPresent?: boolean;
  /** All sub-bundles must match */
  all?: PackTriggers[];
  /** At least one sub-bundle matches */
  any?: PackTriggers[];
};

export type SituationPack = {
  id: string;
  name: string;
  nameHe?: string;
  enabled: boolean;
  triggers: PackTriggers;
  interpretation: string;
  cognition?: string;
  internalState?: Record<string, string | number | boolean>;
  tone: SituationTone;
  priority: SituationPriority;
  cooldownMs: number;
  responses: string[];
  /** Scene builder tags for multi-pack composition */
  sceneTags?: string[];
  proactive: boolean;
};

export type MatchedSituation = {
  pack: SituationPack;
  score: number;
  confidence: number;
};

export type BuiltScene = {
  label: string;
  interpretation: string;
  packIds: string[];
  dominantTone: SituationTone;
};

export type SituationPackDecision = {
  packId: string;
  message: string;
  topic: string;
  mood: "curious" | "excited" | "observing";
  reason: string;
  scene: BuiltScene | null;
  interpretation: string;
  tone: SituationTone;
  priority: SituationPriority;
};

export const PRIORITY_RANK: Record<SituationPriority, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
};
