/** HAL Consciousness — single source of temporal truth. */

export type PresencePhase = "unknown" | "weak" | "stable" | "absent";

export type SoulState =
  | "VOID_IDLE"
  | "PHANTOM_DETECTION"
  | "PRESENCE_FORMING"
  | "STABLE_PRESENCE"
  | "PRESENCE_COLLAPSE";

export type HalGlobalState = {
  presence: {
    phase: PresencePhase;
    soul: SoulState;
    confidence: number;
    stabilitySec: number;
    stableSince: number | null;
    lastSeenAt: number | null;
    lastTransition: string | null;
    lastTransitionAt: number;
  };
  perception: {
    certainty: number;
    ambiguity: number;
    noiseFloor: number;
  };
  worldModel: {
    continuity: number;
    entityCount: number;
  };
  affect: {
    curiosity: number;
    certainty: number;
    alertness: number;
  };
  updatedAt: number;
};

export type SceneMemoryEntry = {
  t: number;
  soul: SoulState;
  transition: string;
  confidence: number;
  rawDetected: boolean;
};

export type PresenceAuthority = {
  /** Authoritative for WorldMemory + Gemma — true only at STABLE_PRESENCE */
  personStable: boolean;
  rawDetected: boolean;
  phase: PresencePhase;
  soul: SoulState;
  confidence: number;
  stabilitySec: number;
  interpretation: string;
  personJustBecameStable: boolean;
  personJustCollapsed: boolean;
};

export type ConsciousnessSnapshot = HalGlobalState & {
  sceneMemory: SceneMemoryEntry[];
  gemmaBlock: string;
  authority: PresenceAuthority;
};

export const SOUL_LABEL_HE: Record<SoulState, string> = {
  VOID_IDLE: "ריק · אין עדות",
  PHANTOM_DETECTION: "פантום · סימנים חלקיים",
  PRESENCE_FORMING: "התהוות · נוכחות מתגבשת",
  STABLE_PRESENCE: "יציב · נוכחות מאושרת",
  PRESENCE_COLLAPSE: "קריסה · נעלם",
};

export const createInitialGlobalState = (now = Date.now()): HalGlobalState => ({
  presence: {
    phase: "absent",
    soul: "VOID_IDLE",
    confidence: 0,
    stabilitySec: 0,
    stableSince: null,
    lastSeenAt: null,
    lastTransition: "init",
    lastTransitionAt: now,
  },
  perception: {
    certainty: 0,
    ambiguity: 1,
    noiseFloor: 0.15,
  },
  worldModel: {
    continuity: 0,
    entityCount: 0,
  },
  affect: {
    curiosity: 0.35,
    certainty: 0,
    alertness: 0.4,
  },
  updatedAt: now,
});
