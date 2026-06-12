/**
 * Camera conversation — separate from text chat (grovee_chats_v1).
 * Own message timeline + durable relationship memory (topics, acquaintance).
 */

import type { CharacterBrainSnapshot } from "./characterBrain";

export const CAMERA_STORAGE_KEY = "grovee_camera_v2";

export type CameraMessageKind = "user" | "reply" | "proactive";

export type CameraMessage = {
  id: string;
  role: "user" | "assistant";
  kind: CameraMessageKind;
  content: string;
  ts: number;
  modelLabel?: string;
  thought?: string;
  visionContext?: string;
};

export type UserProfile = {
  name: string;
  hobbies: string[];
  notes: string;
  updatedAt: number;
};

export type CameraRelationshipMemory = CharacterBrainSnapshot & {
  /** Human-readable topic log for UI (newest last). */
  topicLog: string[];
};

export type CameraSessionStore = {
  version: 2;
  sessionId: string;
  updatedAt: number;
  messages: CameraMessage[];
  memory: CameraRelationshipMemory;
  profile: UserProfile;
  /** Short rolling summary (2–4 lines) for prompt + UI. */
  rollingSummary: string;
};

const emptyProfile = (): UserProfile => ({
  name: "",
  hobbies: [],
  notes: "",
  updatedAt: 0,
});

const emptyMemory = (): CameraRelationshipMemory => ({
  mood: "observing",
  curiosity: 0.35,
  boredom: 0.1,
  acquaintanceDone: false,
  baselineIntroDone: false,
  topicsMentioned: {},
  topicLog: [],
});

export const defaultCameraSessionStore = (): CameraSessionStore => ({
  version: 2,
  sessionId: newCameraSessionId(),
  updatedAt: Date.now(),
  messages: [],
  memory: emptyMemory(),
  profile: emptyProfile(),
  rollingSummary: "",
});

const newCameraSessionId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `cam-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

const migrateV1 = (parsed: Record<string, unknown>): CameraSessionStore => {
  const mem = (parsed.memory ?? {}) as Partial<CameraRelationshipMemory>;
  return {
    version: 2,
    sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : newCameraSessionId(),
    updatedAt: typeof parsed.updatedAt === "number" ? parsed.updatedAt : Date.now(),
    messages: Array.isArray(parsed.messages) ? (parsed.messages as CameraMessage[]) : [],
    memory: {
      ...emptyMemory(),
      ...mem,
      topicsMentioned:
        mem.topicsMentioned && typeof mem.topicsMentioned === "object" ? mem.topicsMentioned : {},
      topicLog: Array.isArray(mem.topicLog) ? mem.topicLog : [],
    },
    profile: emptyProfile(),
    rollingSummary: "",
  };
};

export const loadCameraSessionStore = (): CameraSessionStore => {
  try {
    const raw =
      localStorage.getItem(CAMERA_STORAGE_KEY) ?? localStorage.getItem("grovee_camera_v1");
    if (!raw) return defaultCameraSessionStore();
    const parsed = JSON.parse(raw) as Partial<CameraSessionStore> & { version?: number };
    if (parsed.version === 2) {
      return {
        version: 2,
        sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : newCameraSessionId(),
        updatedAt: typeof parsed.updatedAt === "number" ? parsed.updatedAt : Date.now(),
        messages: Array.isArray(parsed.messages) ? parsed.messages : [],
        memory: {
          ...emptyMemory(),
          ...(parsed.memory ?? {}),
          topicsMentioned:
            parsed.memory?.topicsMentioned && typeof parsed.memory.topicsMentioned === "object"
              ? parsed.memory.topicsMentioned
              : {},
          topicLog: Array.isArray(parsed.memory?.topicLog) ? parsed.memory!.topicLog : [],
        },
        profile: {
          ...emptyProfile(),
          ...(parsed.profile ?? {}),
          hobbies: Array.isArray(parsed.profile?.hobbies) ? parsed.profile!.hobbies : [],
        },
        rollingSummary: typeof parsed.rollingSummary === "string" ? parsed.rollingSummary : "",
      };
    }
    if (parsed.version === 1) return migrateV1(parsed as Record<string, unknown>);
    return defaultCameraSessionStore();
  } catch {
    return defaultCameraSessionStore();
  }
};

export const saveCameraSessionStore = (store: CameraSessionStore): void => {
  try {
    const trimmed: CameraSessionStore = {
      ...store,
      version: 2,
      messages: store.messages.slice(-120).map((m) => ({
        ...m,
        content: m.content.length > 8000 ? `${m.content.slice(0, 8000)}…` : m.content,
      })),
    };
    localStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify(trimmed));
  } catch {
    // quota — skip
  }
};

export const cameraMessageToChatTurn = (
  m: CameraMessage,
): { role: "user" | "assistant"; content: string } => ({
  role: m.role,
  content: m.content,
});

export const buildCameraHistoryForWorker = (
  messages: CameraMessage[],
): Array<{ role: "user" | "assistant"; content: string }> =>
  messages.map(cameraMessageToChatTurn);

export const appendTopicToLog = (log: string[], topic: string, max = 24): string[] => {
  const t = topic.trim();
  if (!t) return log;
  const next = log.filter((x) => x !== t);
  next.push(t);
  return next.slice(-max);
};

export const clearCameraSessionStore = (): CameraSessionStore => {
  const fresh = defaultCameraSessionStore();
  saveCameraSessionStore(fresh);
  return fresh;
};
