/** Vision 2.0 — typed contracts (LLM sees DialogueContext only). */

import type { CharacterMood } from "../characterBrain";

export type PostureKind = "sitting" | "standing" | "unknown";
export type AttentionKind = "screen" | "camera" | "away" | "internal" | "unknown";
export type ActivityKind = "working" | "resting" | "social" | "thinking" | "unknown";
export type EnergyLevel = "low" | "medium" | "high";
export type EnvironmentKind = "office" | "kitchen" | "living" | "bedroom" | "unknown";

export type SituationPrimary =
  | "working"
  | "drinking"
  | "using_phone"
  | "greeting"
  | "reflecting"
  | "idle"
  | "unknown";

export type CoachIntentKind = "offer_support" | "suggest_break" | "encourage" | "none";

export type SocialAwarenessVector = {
  greeting: number;
  interest: number;
  agreement: number;
  disagreement: number;
  confusion: number;
  updatedAt: number;
};

export type ProductivityState = {
  workSessionMin: number;
  focusStreakMin: number;
  breaksTaken: number;
  fatigueLevel: number;
  needsBreak: boolean;
  updatedAt: number;
};

export type TeachingState = {
  attentionLoss: number;
  engagementDrop: number;
  likelyDistracted: boolean;
  ageSec: number;
  updatedAt: number;
};

export type EmotionalState = {
  supportNeeded: number;
  encouragementOpportunity: number;
  motivation: number;
  reflection: number;
  updatedAt: number;
};

export type AudioContext = {
  available: boolean;
  level: number;
  speechDetected: boolean;
};

export type CapabilityContext = {
  social: SocialAwarenessVector;
  productivity: ProductivityState;
  teaching: TeachingState;
  emotional: EmotionalState;
};

export type RecentChangeKind =
  | "entered"
  | "left"
  | "shifted_focus"
  | "stress_rising"
  | "break_needed"
  | "greeting"
  | "activity_change";

/** L2 — factual observations (single frame). */
export type Observation = {
  id: string;
  confidence: number;
};

export type ObservationSet = {
  timestamp: number;
  personPresent: boolean;
  touchingFace: boolean;
  touchingHead: boolean;
  handsOnHead: boolean;
  handNearEyes: boolean;
  handOnChin: boolean;
  raisedHand: boolean;
  waving: boolean;
  pointing: boolean;
  thumbsUp: boolean;
  thumbsDown: boolean;
  holdingCup: boolean;
  usingPhone: boolean;
  gazeDown: boolean;
  gazeAtCamera: boolean;
  motionLevel: number;
};

/** L3 — human state (smoothed). */
export type HumanState = {
  posture: PostureKind;
  attention: AttentionKind;
  activity: ActivityKind;
  energy: EnergyLevel;
  engagement: number;
  updatedAt: number;
};

/** L4 — body language vector. */
export type BodyLanguageVector = {
  focused: number;
  thinking: number;
  stressed: number;
  bored: number;
  ageSec: number;
  updatedAt: number;
};

/** L5 — situation. */
export type SituationState = {
  primary: SituationPrimary;
  confidence: number;
  description: string;
  updatedAt: number;
};

/** L6 — coach. */
export type CoachState = {
  intent: CoachIntentKind;
  reason: string;
  urgency: number;
};

/** L7 — episodic entry. */
export type EpisodicEntryKind = "focus_block" | "stress_episode" | "break" | "greeting" | "face_touch";

export type EpisodicEntry = {
  kind: EpisodicEntryKind;
  startedAt: number;
  durationSec: number;
  peakScores?: Partial<BodyLanguageVector>;
};

/** L8 — world model. */
export type RoomModel = {
  hasLaptop: boolean;
  hasCup: boolean;
  hasPhone: boolean;
  hasTv: boolean;
  environment: EnvironmentKind;
  stableObjects: string[];
  semanticNotes: string;
};

export type PersonModel = {
  present: boolean;
  absentDurationSec: number;
  posture: PostureKind;
  attention: AttentionKind;
  activity: ActivityKind;
  reflecting: boolean;
};

export type SessionModel = {
  sceneAgeSec: number;
  workSessionMin: number;
  lastBreakAt: number;
  lastGreetingAt: number;
  faceTouchDurationSec: number;
};

export type WorldSnapshot = {
  room: RoomModel;
  person: PersonModel;
  session: SessionModel;
  bodyLanguage: BodyLanguageVector;
  situation: SituationState;
  updatedAt: number;
};

export type RecentChange = {
  kind: RecentChangeKind;
  ageSec: number;
  confidence: number;
  text: string;
};

export type HalLayer = {
  mood: CharacterMood;
  tone: string;
  moodLabelHe: string;
  personPresent: boolean;
  sceneLabel: string | null;
  activePackId: string | null;
  interpretation: string | null;
  stressLevel: number;
  engagement: number;
};

/** L9 — fused scene + narrative (Gemma input). */
export type InterpretationLayer = {
  sceneState: {
    activity: string;
    intensity: number;
    stability: string;
    focusTarget: string;
    engagementLevel: number;
  };
  metaEvents: Array<{
    type: string;
    confidence: number;
    components: string[];
    meaning: string;
  }>;
  agentState: {
    curiosity: number;
    attentiveness: number;
    verbosity: number;
    emotionalBias: number;
  };
  narrative: {
    whatChanged: string[];
    interpretation: string[];
    contextShift: string;
    suggestedResponseTone: string;
  };
  /** Full block for LLM — includes internal monologue section. */
  gemmaBlock: string;
};

/** HAL Consciousness — temporal soul state (single source of truth for presence). */
export type ConsciousnessLayer = {
  soul: import("./consciousness/types").SoulState;
  phase: import("./consciousness/types").PresencePhase;
  confidence: number;
  stabilitySec: number;
  personStable: boolean;
  rawDetected: boolean;
  interpretation: string;
  evolution: string;
  affect: import("./consciousness/types").HalGlobalState["affect"];
  perception: import("./consciousness/types").HalGlobalState["perception"];
  gemmaBlock: string;
};

/** L10 — sole LLM input for camera dialogue. */
export type DialogueContext = {
  worldState: {
    room: RoomModel;
    person: Pick<PersonModel, "present" | "absentDurationSec">;
    session: SessionModel;
  };
  personState: HumanState;
  bodyLanguage: BodyLanguageVector;
  situation: SituationState;
  recentChanges: RecentChange[];
  coach: CoachState;
  capabilities: CapabilityContext;
  audio: AudioContext;
  character: {
    mood: CharacterMood;
    shouldSpeak: boolean;
    speakReason?: string;
  };
  hal: HalLayer;
  interpretation: InterpretationLayer;
  consciousness?: ConsciousnessLayer | null;
  episodicSummary: string[];
  /** Who HAL sees — age/gender/engagement (face model, smoothed). */
  entity?: import("./entityProfile").EntityProfile | null;
};

export const EMPTY_BODY_LANGUAGE: BodyLanguageVector = {
  focused: 0,
  thinking: 0,
  stressed: 0,
  bored: 0,
  ageSec: 0,
  updatedAt: 0,
};

export const EMPTY_SITUATION: SituationState = {
  primary: "unknown",
  confidence: 0,
  description: "No person confirmed in frame.",
  updatedAt: 0,
};

export const EMPTY_COACH: CoachState = {
  intent: "none",
  reason: "",
  urgency: 0,
};
