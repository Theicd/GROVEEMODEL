import type { MotionSnapshot } from "./motionLayer";
import { isSituationSubject } from "./sensorBlock";

/** Structured world state — not raw image descriptions. */

export type SemanticEventType =
  | "person_entered"
  | "person_left"
  | "object_appeared"
  | "object_removed"
  | "door_opened"
  | "user_returned"
  | "activity_change"
  | "generic";

export type SemanticEvent = {
  id: string;
  ts: number;
  type: SemanticEventType;
  subject?: string;
  text: string;
  significant: boolean;
};

export type WorldInspectorSnapshot = {
  objects: string[];
  personPresent: boolean;
  poseState: string;
  gestures: string[];
  holding: string[];
  fingerStates: WorldMemory["fingerStates"];
  faceSummary: string;
  emotionEstimate: string;
  lastVisionFrameAt: number;
  memoryAgeSec: number;
  bootContext: string;
  liveContext: string;
};

export type VisionPayload = {
  objects?: string[];
  people?: string[];
  current?: string[];
  events?: string[];
  interesting?: boolean;
  summary?: string;
};

export type LightDetectionPayload = {
  objects: string[];
  people: string[];
};

export type WorldUpdateResult = {
  newEvents: SemanticEvent[];
  appeared: string[];
  removed: string[];
  /** First scan — objects stored as baseline, not events. */
  isBaselineCapture?: boolean;
  /** >50% labels swapped — likely camera pan, not real events. */
  suppressedAsChurn?: boolean;
};

const MAX_EVENTS = 32;
const EVENT_TTL_MS = 15 * 60 * 1000;
const MOTION_EVENT_DEBOUNCE_MS = 12_000;
const MOTION_BURST_DEBOUNCE_MS = 35_000;

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `ev-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

export const normalizeLabel = (s: string): string =>
  s
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");

export const normalizeList = (items: string[]): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of items) {
    const n = normalizeLabel(raw);
    if (!n || seen.has(n)) continue;
    seen.add(n);
    out.push(n);
  }
  return out.slice(0, 16);
};

const classifyEventText = (text: string): SemanticEventType => {
  const t = text.toLowerCase();
  if (/person entered|someone entered|new person|user returned|user came back/i.test(t)) {
    return /return|came back/i.test(t) ? "user_returned" : "person_entered";
  }
  if (/person left|someone left|person removed/i.test(t)) return "person_left";
  if (/door opened|door open/i.test(t)) return "door_opened";
  if (/removed|disappeared|gone|left desk|no longer/i.test(t)) return "object_removed";
  if (/appeared|added|new object|picked up|entered frame/i.test(t)) return "object_appeared";
  if (/working|using laptop|typing|phone/i.test(t)) return "activity_change";
  return "generic";
};

const isSignificantType = (type: SemanticEventType): boolean =>
  type === "person_entered" ||
  type === "person_left" ||
  type === "object_appeared" ||
  type === "object_removed" ||
  type === "door_opened" ||
  type === "user_returned";

/** Objects worth a deep Gemma pass when newly detected (Stage 2). */
export const SIGNIFICANT_LIGHT_OBJECTS = new Set([
  "phone",
  "laptop",
  "television",
  "book",
  "keyboard",
  "backpack",
]);

const isLikelyCameraChurn = (prev: string[], next: string[]): boolean => {
  if (prev.length === 0) return false;
  const appeared = next.filter((o) => !prev.includes(o));
  const removed = prev.filter((o) => !next.includes(o));
  const churn = (appeared.length + removed.length) / Math.max(1, prev.length);
  return churn >= 0.55 && appeared.length >= 2 && removed.length >= 1;
};

const personEventType = (world: WorldMemory, prevHadPerson: boolean): SemanticEventType => {
  if (!prevHadPerson) return "person_entered";
  const recentLeave = world.lastChanges.some(
    (e) => e.type === "person_left" && Date.now() - e.ts < 120_000,
  );
  return recentLeave ? "user_returned" : "person_entered";
};

/** Skip generic vision "discovery" lines for objects already in baseline. */
const isBaselineNoiseEvent = (text: string, baselineObjects: string[]): boolean => {
  const t = text.toLowerCase();
  if (/visible|detected|in frame|in the room|present/i.test(t)) {
    for (const obj of baselineObjects) {
      if (t.includes(obj)) return true;
    }
  }
  return false;
};

export const makeSemanticEvent = (
  type: SemanticEventType,
  text: string,
  subject?: string,
  forceSignificant = false,
): SemanticEvent => ({
  id: newId(),
  ts: Date.now(),
  type,
  subject: subject ? normalizeLabel(subject) : undefined,
  text: text.trim(),
  significant: forceSignificant || isSignificantType(type),
});

/** Parse vision JSON — supports legacy `current` and structured `objects`/`people`. */
export const parseSceneAnalysisJson = (raw: string): VisionPayload | null => {
  const trimmed = raw.trim();
  const jsonMatch = trimmed.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return null;
  try {
    const parsed = JSON.parse(jsonMatch[0]) as Record<string, unknown>;
    const pickStrings = (key: string) => {
      const arr = parsed[key];
      if (!Array.isArray(arr)) return [] as string[];
      return arr.filter((x): x is string => typeof x === "string").map((s) => s.trim()).filter(Boolean);
    };
    const objects = pickStrings("objects");
    const people = pickStrings("people");
    const current = pickStrings("current");
    const events = pickStrings("events");
    const summary = typeof parsed.summary === "string" ? parsed.summary.trim() : "";
    const interesting = parsed.interesting === true;
    return {
      objects: objects.length ? objects : undefined,
      people: people.length ? people : undefined,
      current: current.length ? current : undefined,
      events,
      interesting,
      summary: summary || current.join("; ") || objects.join("; "),
    };
  } catch {
    return null;
  }
};

export class WorldMemory {
  objects: string[] = [];
  people: string[] = [];
  baselineObjects: string[] = [];
  baselinePeople: string[] = [];
  baselineEstablished = false;
  lastSummary = "";
  lastChanges: SemanticEvent[] = [];
  sceneStartedAt = 0;
  lastChangeAt = 0;
  lastAnalysisAt = 0;
  lastSnapshotAt = 0;
  /** Debounced person presence (from COCO). */
  personPresent = false;
  absentSince = 0;
  /** Duration of last absence before re-entry (for tiered welcome). */
  lastAbsentDurationMs = 0;
  lastMotionLevel = 0;
  lastMotionAt = 0;
  private lastMotionEventAt = new Map<string, number>();
  /** Pose / situation layer (MoveNet + heuristics). */
  poseState: "standing" | "sitting" | "unknown" = "unknown";
  poseConfidence = 0;
  poseUpdatedAt = 0;
  poseSource = "none";
  gestures: string[] = [];
  holding: string[] = [];
  focusHint = "";
  /** Latest situation-layer subject (wave, object_held:cup, stood_with_drink…). */
  lastSituationSubject = "";
  lastSituationAt = 0;
  /** Rule-based scene text from vision-lab (replaces frequent Gemma analyze_scene). */
  richSceneDescription = "";
  environment = "";
  labBodyLanguage: string[] = [];
  /** Tentative emotion estimate — HAL must not state as fact. */
  emotionEstimate = "";
  /** Latest finger counts per hand from vision-lab. */
  fingerStates: Array<{
    hand: "Left" | "Right";
    count: number;
    thumb: string;
    index: string;
    middle: string;
    ring: string;
    pinky: string;
  }> = [];
  /** Short face summary for chat context (estimate only). */
  faceSummary = "";
  /** Timestamp of last vision-lab frame synced into memory. */
  lastVisionFrameAt = 0;
  /** Deep Gemma boot snapshot — room baseline (set once). */
  bootContext = "";
  /** Live rolling context from small models — updated every vision frame. */
  liveContext = "";

  get sceneAgeSec(): number {
    if (!this.sceneStartedAt) return 0;
    return Math.floor((Date.now() - this.sceneStartedAt) / 1000);
  }

  toInspectorSnapshot(): WorldInspectorSnapshot {
    return {
      objects: [...this.objects],
      personPresent: this.personPresent,
      poseState: this.poseState,
      gestures: [...this.gestures],
      holding: [...this.holding],
      fingerStates: this.fingerStates.map((f) => ({ ...f })),
      faceSummary: this.faceSummary,
      emotionEstimate: this.emotionEstimate,
      lastVisionFrameAt: this.lastVisionFrameAt,
      memoryAgeSec: this.lastVisionFrameAt
        ? Math.max(0, Math.floor((Date.now() - this.lastVisionFrameAt) / 1000))
        : 0,
      bootContext: this.bootContext,
      liveContext: this.liveContext,
    };
  }

  reset(): void {
    this.objects = [];
    this.people = [];
    this.baselineObjects = [];
    this.baselinePeople = [];
    this.baselineEstablished = false;
    this.lastSummary = "";
    this.lastChanges = [];
    this.sceneStartedAt = 0;
    this.lastChangeAt = 0;
    this.lastAnalysisAt = 0;
    this.lastSnapshotAt = 0;
    this.personPresent = false;
    this.absentSince = 0;
    this.lastAbsentDurationMs = 0;
    this.lastMotionLevel = 0;
    this.lastMotionAt = 0;
    this.lastMotionEventAt.clear();
    this.poseState = "unknown";
    this.poseConfidence = 0;
    this.poseUpdatedAt = 0;
    this.poseSource = "none";
    this.gestures = [];
    this.holding = [];
    this.focusHint = "";
    this.lastSituationSubject = "";
    this.lastSituationAt = 0;
    this.richSceneDescription = "";
    this.environment = "";
    this.labBodyLanguage = [];
    this.emotionEstimate = "";
    this.fingerStates = [];
    this.faceSummary = "";
    this.lastVisionFrameAt = 0;
    this.bootContext = "";
    this.liveContext = "";
  }

  /** Append semantic events from vision-lab rule engine. */
  applySemanticEvents(events: SemanticEvent[]): void {
    if (events.length) this.recordEvents(events, Date.now());
  }

  msSinceAbsent(): number {
    if (!this.absentSince) return 0;
    return Date.now() - this.absentSince;
  }

  private canEmitMotion(kind: string): boolean {
    const last = this.lastMotionEventAt.get(kind) ?? 0;
    const windowMs = kind === "burst" ? MOTION_BURST_DEBOUNCE_MS : MOTION_EVENT_DEBOUNCE_MS;
    if (Date.now() - last < windowMs) return false;
    this.lastMotionEventAt.set(kind, Date.now());
    return true;
  }

  /** Stage 1 — motion diff between consecutive frames (~3s). */
  applyMotion(snap: MotionSnapshot, personInFrame: boolean): WorldUpdateResult {
    const now = Date.now();
    this.lastMotionLevel = snap.motionLevel;
    this.lastMotionAt = now;
    this.personPresent = personInFrame;

    const newEvents: SemanticEvent[] = [];

    if (snap.waveLike && personInFrame && this.canEmitMotion("wave")) {
      newEvents.push(
        makeSemanticEvent("activity_change", "Person waving or gesturing at camera", "wave", true),
      );
    } else if (snap.armMovement && snap.motionLevel >= 0.14 && personInFrame && this.canEmitMotion("arm")) {
      newEvents.push(
        makeSemanticEvent("activity_change", "Arm movement in upper frame", "arm_movement", true),
      );
    } else if (snap.motionLevel >= 0.32 && personInFrame && this.canEmitMotion("burst")) {
      newEvents.push(
        makeSemanticEvent("activity_change", "Sudden high motion burst", "motion_burst", true),
      );
    }

    if (newEvents.length) this.recordEvents(newEvents, now);
    return { newEvents, appeared: [], removed: [] };
  }

  hasData(): boolean {
    return (
      this.objects.length > 0 ||
      this.people.length > 0 ||
      this.lastChanges.length > 0 ||
      !!this.lastSummary.trim()
    );
  }

  /** Enough context for CharacterBrain without Gemma deep vision (COCO + motion + events). */
  hasSensorContext(): boolean {
    if (!this.baselineEstablished) return false;
    return (
      this.hasData() ||
      this.lastMotionAt > 0 ||
      this.sceneStartedAt > 0
    );
  }

  clearPersonLayer(): void {
    this.poseState = "unknown";
    this.poseConfidence = 0;
    this.poseUpdatedAt = 0;
    this.poseSource = "none";
    this.gestures = [];
    this.holding = [];
    this.focusHint = "";
    this.fingerStates = [];
    this.faceSummary = "";
  }

  /** When Gemma deep vision is skipped/failed — lightweight HAL still gets a summary line. */
  applySensorBaseline(sensorBlock: string): void {
    const hint = sensorBlock.trim().slice(0, 280);
    if (!this.lastSummary.trim()) {
      this.lastSummary = hint
        ? `Lightweight scene watch: ${hint.replace(/\s+/g, " ").slice(0, 120)}`
        : "Lightweight scene watch: monitoring via motion and object detection.";
    }
    this.lastAnalysisAt = Date.now();
  }

  /** Drop person-only activity events when nobody is in frame (avoids stale wave/pose noise). */
  pruneStalePersonActivity(): void {
    if (this.personPresent) return;
    const cutoff = Date.now() - 45_000;
    this.lastChanges = this.lastChanges.filter((e) => {
      if (e.ts < cutoff) return false;
      if (e.type !== "activity_change") return true;
      const sub = e.subject ?? "";
      if (/^(wave|arm_movement|motion_burst|pose_change|stood_with_drink|object_held:)/.test(sub)) {
        return false;
      }
      return true;
    });
    if (this.lastSituationSubject && Date.now() - this.lastSituationAt > 60_000) {
      this.lastSituationSubject = "";
      this.lastSituationAt = 0;
    }
  }

  msSinceLastChange(): number {
    if (!this.lastChangeAt) return Number.POSITIVE_INFINITY;
    return Date.now() - this.lastChangeAt;
  }

  /** Stage 2 — COCO-SSD updates (primary scene state + events). */
  applyLightDetection(payload: LightDetectionPayload): WorldUpdateResult {
    const now = Date.now();
    if (!this.sceneStartedAt) this.sceneStartedAt = now;
    this.lastSnapshotAt = now;

    const prevObjects = [...this.objects];
    const prevHadPerson = this.people.length > 0;

    const nextObjects = normalizeList(payload.objects);
    const nextPeople = normalizeList(payload.people);

    if (!this.baselineEstablished) {
      this.objects = nextObjects;
      this.people = nextPeople;
      this.baselineEstablished = true;
      this.baselineObjects = [...nextObjects];
      this.baselinePeople = [...nextPeople];
      return {
        newEvents: [],
        appeared: [],
        removed: [],
        isBaselineCapture: true,
      };
    }

    if (isLikelyCameraChurn(prevObjects, nextObjects)) {
      this.objects = nextObjects;
      this.people = nextPeople;
      return {
        newEvents: [],
        appeared: [],
        removed: [],
        suppressedAsChurn: true,
      };
    }

    this.objects = nextObjects;
    const nextHadPerson = nextPeople.length > 0;
    this.people = nextPeople;
    this.personPresent = nextHadPerson;

    if (prevHadPerson && !nextHadPerson) {
      this.absentSince = now;
    } else if (!prevHadPerson && nextHadPerson) {
      this.lastAbsentDurationMs = this.absentSince ? now - this.absentSince : 0;
      this.absentSince = 0;
    }

    const appeared = nextObjects.filter((o) => !prevObjects.includes(o));
    const removed = prevObjects.filter((o) => !nextObjects.includes(o));
    const newEvents: SemanticEvent[] = [];

    if (!prevHadPerson && nextHadPerson) {
      newEvents.push(
        makeSemanticEvent(
          personEventType(this, prevHadPerson),
          "Person entered frame",
          "person",
        ),
      );
    } else if (prevHadPerson && !nextHadPerson) {
      newEvents.push(makeSemanticEvent("person_left", "Person left frame", "person"));
    }

    for (const o of appeared) {
      if (this.baselineObjects.includes(o)) continue;
      if (/door/i.test(o)) {
        newEvents.push(makeSemanticEvent("door_opened", "Door opened", o));
      } else {
        newEvents.push(makeSemanticEvent("object_appeared", `New object: ${o}`, o));
      }
    }
    for (const o of removed) {
      if (this.baselineObjects.includes(o)) continue;
      newEvents.push(makeSemanticEvent("object_removed", `Object removed: ${o}`, o));
    }

    this.recordEvents(newEvents, now);
    return { newEvents, appeared, removed };
  }

  /** Stage 3 — Gemma enriches summary + non-COCO objects only (no people/events). */
  applyDeepVision(payload: VisionPayload): void {
    this.lastAnalysisAt = Date.now();
    if (payload.summary?.trim()) {
      this.bootContext = payload.summary.trim().slice(0, 520);
      if (!this.liveContext.trim()) {
        this.lastSummary = this.bootContext;
      }
    }
    const extra = normalizeList([...(payload.objects ?? []), ...(payload.current ?? [])]);
    if (extra.length) {
      this.objects = normalizeList([...this.objects, ...extra]);
    }
  }

  hasSignificantLightEvent(result: WorldUpdateResult): boolean {
    return result.newEvents.some(
      (e) =>
        e.type === "person_entered" ||
        e.type === "user_returned" ||
        (e.type === "object_appeared" &&
          !!e.subject &&
          SIGNIFICANT_LIGHT_OBJECTS.has(e.subject)),
    );
  }

  private recordEvents(newEvents: SemanticEvent[], now: number): void {
    if (!newEvents.length) return;
    for (const ev of newEvents) {
      if (ev.type === "activity_change" && ev.subject && isSituationSubject(ev.subject)) {
        this.lastSituationSubject = ev.subject;
        this.lastSituationAt = now;
      }
    }
    this.lastChangeAt = now;
    this.lastChanges = [...newEvents, ...this.lastChanges]
      .filter((e) => now - e.ts < EVENT_TTL_MS)
      .slice(0, MAX_EVENTS);
  }

  /** Exposed for situationEngine pose events. */
  recordPublicEvents(newEvents: SemanticEvent[]): void {
    if (!newEvents.length) return;
    this.recordEvents(newEvents, Date.now());
  }

  /** @deprecated Prefer applyLightDetection + applyDeepVision. Kept for tests. */
  applyVision(payload: VisionPayload): WorldUpdateResult {
    const light: LightDetectionPayload = {
      objects: payload.objects?.length
        ? payload.objects
        : payload.current?.length
          ? payload.current
          : [],
      people: payload.people ?? [],
    };
    const result = this.applyLightDetection(light);
    if (payload.summary?.trim() || payload.events?.length) {
      if (result.isBaselineCapture) {
        if (payload.summary?.trim()) this.lastSummary = payload.summary.trim();
        return result;
      }
      const prevEvents = result.newEvents.length;
      for (const raw of payload.events ?? []) {
        const text = raw.trim();
        if (!text || isBaselineNoiseEvent(text, this.baselineObjects)) continue;
        const type = classifyEventText(text);
        const subject = text.match(/:\s*(.+)$/)?.[1]?.trim();
        const ev = makeSemanticEvent(type, text, subject);
        if (ev.type === "object_appeared" && ev.subject && this.baselineObjects.includes(ev.subject)) {
          continue;
        }
        if (!result.newEvents.some((e) => e.text.toLowerCase() === ev.text.toLowerCase())) {
          result.newEvents.push(ev);
        }
      }
      if (payload.summary?.trim()) this.lastSummary = payload.summary.trim();
      if (result.newEvents.length > prevEvents) {
        this.recordEvents(result.newEvents.slice(prevEvents), Date.now());
      }
    }
    return result;
  }

  toPromptBlock(maxAgeMs = 5 * 60 * 1000): string {
    const now = Date.now();
    const lines: string[] = [];
    lines.push(`World memory (scene age ${this.sceneAgeSec}s):`);
    if (this.objects.length) lines.push(`Objects: ${this.objects.join(", ")}`);
    if (this.people.length) lines.push(`People: ${this.people.join(", ")}`);
    if (this.lastSummary) lines.push(`Summary: ${this.lastSummary}`);

    const recent = this.lastChanges.filter((e) => now - e.ts <= maxAgeMs);
    if (recent.length) {
      lines.push("Recent events:");
      for (const e of recent.slice(0, 10)) {
        const ageSec = Math.round((now - e.ts) / 1000);
        lines.push(`- [${e.type}] ${e.text} (${ageSec}s ago)`);
      }
    }
    lines.push(
      "You are a present AI character — not a security camera. Use memory; do not re-describe unchanged things.",
    );
    return lines.join("\n");
  }

  /** For chat — atmosphere only; posture comes from FRESH PERSON block on demand. */
  toCharacterAtmosphereBlock(maxAgeMs = 5 * 60 * 1000): string {
    const now = Date.now();
    const lines: string[] = [];
    lines.push(`Scene context (age ${this.sceneAgeSec}s):`);
    if (this.people.length) lines.push(`People detected: ${this.people.join(", ")}`);
    else lines.push("People detected: none");
    if (this.lastSummary) lines.push(`Atmosphere: ${this.lastSummary}`);
    const recent = this.lastChanges.filter(
      (e) => now - e.ts <= maxAgeMs && !/^pose_change:/.test(e.subject ?? ""),
    );
    if (recent.length) {
      lines.push("Recent changes (not posture — may be stale):");
      for (const e of recent.slice(0, 3)) {
        lines.push(`- ${e.text}`);
      }
    }
    return lines.join("\n");
  }

  /** For chat — atmosphere only; no object inventory that triggers captioning. */
  toCharacterChatBlock(maxAgeMs = 5 * 60 * 1000): string {
    const now = Date.now();
    const lines: string[] = [];
    lines.push(`Scene context (age ${this.sceneAgeSec}s):`);
    if (this.people.length) lines.push(`People detected: ${this.people.join(", ")}`);
    else lines.push("People detected: none");
    if (this.poseUpdatedAt) {
      const poseAgeSec = Math.floor((now - this.poseUpdatedAt) / 1000);
      lines.push(
        `Person posture (age ${poseAgeSec}s, confidence ${this.poseConfidence.toFixed(2)}, source ${this.poseSource}): ${this.poseState}`,
      );
      if (poseAgeSec > 15) {
        lines.push("WARNING: posture memory may be stale — prefer FRESH PERSON block if provided.");
      }
    }
    if (this.holding.length) lines.push(`Holding (sensor): ${this.holding.join(", ")}`);
    if (this.lastSummary) lines.push(`Atmosphere: ${this.lastSummary}`);
    const recent = this.lastChanges.filter((e) => now - e.ts <= maxAgeMs);
    if (recent.length) {
      lines.push("Recent changes:");
      for (const e of recent.slice(0, 4)) {
        lines.push(`- ${e.text}`);
      }
    }
    return lines.join("\n");
  }

  toCharacterContext(): string {
    return JSON.stringify(
      {
        objects: this.objects,
        people: this.people,
        personPresent: this.personPresent,
        absentSec: this.absentSince ? Math.floor(this.msSinceAbsent() / 1000) : 0,
        motionLevel: this.lastMotionLevel,
        poseState: this.poseState,
        poseConfidence: this.poseConfidence,
        poseSource: this.poseSource,
        gestures: this.gestures,
        holding: this.holding,
        focusHint: this.focusHint,
        lastChanges: this.lastChanges.slice(0, 8).map((e) => ({ type: e.type, subject: e.subject, text: e.text })),
        sceneAge: this.sceneAgeSec,
      },
      null,
      0,
    );
  }
}

/** @deprecated use WorldMemory — kept for gradual migration */
export type SceneAnalysisPayload = VisionPayload & {
  current: string[];
  events: string[];
  interesting: boolean;
  summary: string;
};

/** @deprecated use WorldMemory */
export class EnvironmentMemory extends WorldMemory {
  get current(): string[] {
    return this.objects;
  }

  applyAnalysis(payload: SceneAnalysisPayload): import("./worldMemory").SemanticEvent[] {
    const result = this.applyVision({
      objects: payload.current,
      events: payload.events,
      interesting: payload.interesting,
      summary: payload.summary,
    });
    return result.newEvents;
  }

  get events() {
    return this.lastChanges;
  }
}

export type SceneEvent = SemanticEvent;
