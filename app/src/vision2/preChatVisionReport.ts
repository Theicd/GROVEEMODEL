/**
 * Pre-chat vision report — sensor fusion for Gemma (not consciousness-only).
 * Chat answers MUST use this report; narrative layer must not override it.
 */

import type { VisionResult } from "../vision-lab/core/types";
import type { WorldMemory } from "../worldMemory";
import { correctAgeForDisplay } from "./entityProfile";
import type { DialogueContext } from "./types";

/** Match Vision Lab HUD (person can show at ~0.55–0.65). */
export const PERSON_DETECT_MIN_CONF = 0.35;

export type PreChatVisionReport = {
  personVisible: boolean;
  yoloPersonCount: number;
  faceCount: number;
  maxYoloConfidence: number;
  /** System prompt block for the model (low-echo format). */
  text: string;
  /** English perception summary for UI "thinking" panel — never shown as chat answer. */
  internalEn: string;
};

export const countYoloPersons = (vision: VisionResult | null): { count: number; maxConf: number } => {
  if (!vision?.objects.length) return { count: 0, maxConf: 0 };
  const persons = vision.objects.filter(
    (o) => o.label === "person" && o.confidence >= PERSON_DETECT_MIN_CONF,
  );
  return {
    count: persons.length,
    maxConf: persons.reduce((m, o) => Math.max(m, o.confidence), 0),
  };
};

/** Sensors → "is someone visible for chat?" (not STABLE-only). */
export const resolvePersonVisibleForChat = (params: {
  vision: VisionResult | null;
  dialogue: DialogueContext | null;
  world: WorldMemory;
}): boolean => {
  const { vision, dialogue, world } = params;
  const { count } = countYoloPersons(vision);
  if (count > 0) return true;
  if ((vision?.faces.length ?? 0) > 0) return true;
  if (dialogue?.consciousness?.rawDetected) return true;
  if (dialogue?.consciousness?.personStable) return true;
  if (world.personPresent) return true;
  return false;
};

const genderEn = (g: string): string => {
  const x = g.toLowerCase();
  if (x.includes("female") || x.includes("woman")) return "female (estimate)";
  if (x.includes("male") && !x.includes("female")) return "male (estimate)";
  return "unclear";
};

type VisionFacts = {
  personVisible: boolean;
  yoloPersonCount: number;
  faceCount: number;
  maxYoloConfidence: number;
  snapshotAttached: boolean;
  holding: string[];
  gestures: string[];
  pose: string;
  emotion: string;
  emotionScore: number;
  attention: string;
  activity: string;
  engagement: number;
  touchingFace: boolean;
  touchingHead: boolean;
  holdingCup: boolean;
  scene: string;
  faces: Array<{ gender: string; age: number; gaze: string }>;
  soul: string;
  soulStable: boolean;
};

const collectVisionFacts = (params: {
  vision: VisionResult | null;
  dialogue: DialogueContext | null;
  world: WorldMemory;
  snapshotAttached: boolean;
}): VisionFacts => {
  const { vision, dialogue, world, snapshotAttached } = params;
  const { count: yoloPersonCount, maxConf: maxYoloConfidence } = countYoloPersons(vision);
  const faceCount = vision?.faces.length ?? 0;
  const personVisible = resolvePersonVisibleForChat({ vision, dialogue, world });
  const p = dialogue?.personState;
  const c = dialogue?.consciousness;
  const bodySignals = (vision?.bodyLanguage ?? []).map((x) => x.signal.toLowerCase());
  const bodyLang = dialogue?.bodyLanguage;

  const faces =
    vision?.faces?.slice(0, 3).map((f) => ({
      gender: genderEn(f.estimatedGender),
      age: correctAgeForDisplay(f.estimatedAge),
      gaze: f.gazeDirection,
    })) ?? [];

  return {
    personVisible,
    yoloPersonCount,
    faceCount,
    maxYoloConfidence,
    snapshotAttached,
    holding: world.holding.length ? [...world.holding] : [],
    gestures: world.gestures.length ? [...world.gestures] : [],
    pose: world.poseState && world.poseState !== "unknown" ? world.poseState : "unknown",
    emotion: vision?.emotion?.dominant ?? "unknown",
    emotionScore: vision?.emotion?.dominantScore ?? 0,
    attention: p?.attention ?? "unknown",
    activity: p?.activity ?? "unknown",
    engagement: p?.engagement ?? 0,
    touchingFace:
      bodySignals.some((s) => /hand on face|chin|jaw/.test(s)) ||
      (bodyLang?.thinking ?? 0) >= 0.5,
    touchingHead: bodySignals.some((s) => /hand on head|hands on head/.test(s)),
    holdingCup: (vision?.interactions ?? []).some((i) => /holding cup/i.test(i.name)),
    scene: vision?.sceneDescription?.trim().slice(0, 280) ?? "",
    faces,
    soul: c?.soul ?? "unknown",
    soulStable: !!c?.personStable,
  };
};

/** English-only summary for the chat "thinking" panel. */
export const buildInternalVisionContextEn = (params: {
  vision: VisionResult | null;
  dialogue: DialogueContext | null;
  world: WorldMemory;
  cameraActive: boolean;
  snapshotAttached?: boolean;
}): string => {
  const { cameraActive, snapshotAttached } = params;
  if (!cameraActive) return "Camera: OFF";

  const f = collectVisionFacts({ ...params, snapshotAttached: !!snapshotAttached });
  const lines: string[] = [
    "Perception snapshot (internal — not for user display):",
    `Person visible: ${f.personVisible ? "YES" : "NO"}`,
    `YOLO persons: ${f.yoloPersonCount}${f.maxYoloConfidence ? ` (max ${(f.maxYoloConfidence * 100).toFixed(0)}%)` : ""}`,
    `Faces: ${f.faceCount}`,
    `Fresh snapshot: ${f.snapshotAttached ? "yes" : "no"}`,
  ];

  if (f.faces.length) {
    lines.push("Face estimates:");
    for (let i = 0; i < f.faces.length; i++) {
      const face = f.faces[i];
      lines.push(`  · #${i + 1}: ${face.gender}, age ~${face.age}, gaze ${face.gaze}`);
    }
  }
  if (f.holding.length) lines.push(`Holding (sensor): ${f.holding.join(", ")}`);
  else lines.push("Holding (sensor): none detected");
  if (f.gestures.length) lines.push(`Gestures: ${f.gestures.join(", ")}`);
  if (f.pose !== "unknown") lines.push(`Posture: ${f.pose}`);
  if (f.emotion !== "unknown") {
    lines.push(`Emotion: ${f.emotion} (${Math.round(f.emotionScore * 100)}%)`);
  }
  if (f.touchingFace) lines.push("Body language: hand on face / chin (thinking or self-touch)");
  if (f.touchingHead) lines.push("Body language: hands on head (stress signal)");
  if (f.holdingCup) lines.push("Interaction: holding cup");
  if (f.attention !== "unknown" || f.activity !== "unknown") {
    lines.push(
      `Attention: ${f.attention} | Activity: ${f.activity} | Engagement ${(f.engagement * 100).toFixed(0)}%`,
    );
  }
  if (f.scene) lines.push(`Scene (VLM): ${f.scene}`);
  lines.push(`HAL soul: ${f.soul} | stable presence: ${f.soulStable ? "yes" : "no"}`);

  return lines.join("\n");
};

export const buildPreChatVisionReport = (params: {
  vision: VisionResult | null;
  dialogue: DialogueContext | null;
  world: WorldMemory;
  cameraActive: boolean;
  snapshotAttached?: boolean;
}): PreChatVisionReport => {
  const { cameraActive, snapshotAttached } = params;

  if (!cameraActive) {
    return {
      personVisible: false,
      yoloPersonCount: 0,
      faceCount: 0,
      maxYoloConfidence: 0,
      text: "[INTERNAL VISION CONTEXT]\nCamera: OFF\n[/INTERNAL]",
      internalEn: "Camera: OFF",
    };
  }

  const f = collectVisionFacts({ ...params, snapshotAttached: !!snapshotAttached });
  const internalEn = buildInternalVisionContextEn(params);

  const modelLines: string[] = [
    "[INTERNAL VISION CONTEXT — use for answers; NEVER repeat this block to the user]",
    `personVisible=${f.personVisible ? "yes" : "no"}`,
    `yoloPersons=${f.yoloPersonCount}${f.maxYoloConfidence ? ` maxConf=${(f.maxYoloConfidence * 100).toFixed(0)}%` : ""}`,
    `faces=${f.faceCount}`,
    `snapshotAttached=${f.snapshotAttached ? "yes" : "no"}`,
  ];

  if (f.faces.length) {
    modelLines.push("faceData:");
    for (let i = 0; i < f.faces.length; i++) {
      const face = f.faces[i];
      modelLines.push(`  person${i + 1}: ${face.gender}, age~${face.age}, gaze=${face.gaze}`);
    }
  }
  if (f.holding.length) modelLines.push(`holding=${f.holding.join(",")}`);
  if (f.gestures.length) modelLines.push(`gestures=${f.gestures.join(",")}`);
  if (f.pose !== "unknown") modelLines.push(`posture=${f.pose}`);
  if (f.emotion !== "unknown") {
    modelLines.push(`emotion=${f.emotion} ${Math.round(f.emotionScore * 100)}%`);
  }
  if (f.touchingFace) modelLines.push("selfTouch=hand_on_face");
  if (f.touchingHead) modelLines.push("selfTouch=hands_on_head");
  if (f.attention !== "unknown" || f.activity !== "unknown") {
    modelLines.push(`attention=${f.attention} activity=${f.activity} engagement=${(f.engagement * 100).toFixed(0)}%`);
  }
  if (f.scene) modelLines.push(`sceneBrief=${f.scene}`);
  modelLines.push(
    "Reply in Hebrew, 2-4 sentences, concrete facts. Do NOT output this internal block.",
    "[/INTERNAL]",
  );

  return {
    personVisible: f.personVisible,
    yoloPersonCount: f.yoloPersonCount,
    faceCount: f.faceCount,
    maxYoloConfidence: f.maxYoloConfidence,
    text: modelLines.join("\n"),
    internalEn,
  };
};
