/**
 * Maps browser-vision-lab VisionResult → structured sensor text for Gemma / HAL.
 * Emotion is labeled as estimate-only per GROVEE character rules.
 */

import type { VisionResult } from "./vision-lab/core/types";
import type { WorldMemory } from "./worldMemory";
import { buildSensorBlock, poseFromWorld } from "./sensorBlock";

const EMOTION_DISCLAIMER =
  "Expression (estimate only — do not state as fact):";

export const mapPoseActionToState = (
  actions: VisionResult["poseActions"],
): { poseState: "standing" | "sitting" | "unknown"; confidence: number } => {
  const standing = actions.find((a) => a.name === "Standing");
  const sitting = actions.find((a) => a.name === "Sitting" || a.name === "Lying");
  if (standing && standing.confidence >= 0.7) {
    return { poseState: "standing", confidence: standing.confidence };
  }
  if (sitting && sitting.confidence >= 0.7) {
    return { poseState: "sitting", confidence: sitting.confidence };
  }
  return { poseState: "unknown", confidence: 0 };
};

export const mapLabGestures = (result: VisionResult): string[] => {
  const out: string[] = [];
  for (const g of result.staticGestures) out.push(g.name.toLowerCase().replace(/\s+/g, "_"));
  for (const g of result.motionGestures) out.push(g.name.toLowerCase().replace(/\s+/g, "_"));
  for (const a of result.poseActions) {
    if (/raised/i.test(a.name)) out.push("wave");
  }
  return [...new Set(out)].slice(0, 6);
};

export const mapLabHolding = (result: VisionResult): string[] => {
  const out: string[] = [];
  for (const i of result.interactions) {
    if (/holding cup/i.test(i.name)) out.push("cup");
    if (/phone/i.test(i.name)) out.push("phone");
    if (/laptop/i.test(i.name)) out.push("laptop");
  }
  for (const o of result.objects) {
    if (o.label === "cup" || o.label === "bottle") out.push(o.label);
    if (o.label === "cell phone") out.push("phone");
  }
  return [...new Set(out)].slice(0, 4);
};

/** Rich multi-layer sensor block for chat system prompt when camera is active. */
export const buildRichSensorBlock = (world: WorldMemory, result?: VisionResult | null): string => {
  const lines: string[] = [buildSensorBlock(world, poseFromWorld(world))];

  if (world.bootContext.trim()) {
    lines.push(`Room baseline (boot vision, once): ${world.bootContext.trim()}`);
  }
  if (world.liveContext.trim()) {
    lines.push(`Live state (updated every frame): ${world.liveContext.trim()}`);
  }

  if (result) {
    if (result.sceneDescription?.trim()) {
      lines.push(`Scene (rule-based): ${result.sceneDescription.trim()}`);
    }
    if (result.environment && result.environment !== "Unknown") {
      lines.push(`Environment hint: ${result.environment}`);
    }
    if (result.bodyLanguage.length) {
      const cues = result.bodyLanguage
        .slice(0, 3)
        .map((c) => `${c.signal} → ${c.meaning}`)
        .join(" | ");
      lines.push(`Body language cues: ${cues}`);
    }
    if (result.events.length) {
      lines.push(
        `Active events: ${result.events.map((e) => e.name).join(", ")}`,
      );
    }
    if (result.emotion?.dominant) {
      lines.push(
        `${EMOTION_DISCLAIMER} ${result.emotion.dominant} (${Math.round(result.emotion.dominantScore * 100)}% est.)`,
      );
    }
    if (result.fingerStates.length) {
      const counts = result.fingerStates.map((f) => `${f.hand}:${f.count}`).join(", ");
      lines.push(`Finger counts: ${counts}`);
    }
  } else if (world.richSceneDescription?.trim()) {
    lines.push(`Scene (rule-based): ${world.richSceneDescription.trim()}`);
  }

  if (world.labBodyLanguage.length) {
    lines.push(`Body language (cached): ${world.labBodyLanguage.slice(0, 3).join(" | ")}`);
  }
  if (world.emotionEstimate?.trim()) {
    lines.push(`${EMOTION_DISCLAIMER} ${world.emotionEstimate.trim()}`);
  }

  if (!result && world.fingerStates.length) {
    const counts = world.fingerStates.map((f) => `${f.hand}:${f.count}`).join(", ");
    lines.push(`Finger counts (cached): ${counts}`);
  }
  if (!result && world.faceSummary.trim()) {
    lines.push(`Face (cached estimate): ${world.faceSummary.trim()}`);
  }

  return lines.join("\n");
};

/** Explicit finger-count block for direct factual answers (e.g. "כמה אצבעות?"). */
export const buildFingerCountBlock = (result: VisionResult): string => {
  if (!result.hands.length && !result.fingerStates.length) {
    return "FINGER SENSOR: no hand detected in the latest vision frame. Say you cannot see a clear hand — ask the user to hold their hand closer to the camera.";
  }

  const lines: string[] = ["FRESH FINGER COUNT (answer from this data only):"];
  for (const f of result.fingerStates) {
    lines.push(
      `${f.hand} hand: ${f.count} finger(s) extended (thumb=${f.fingers.thumb}, index=${f.fingers.index}, middle=${f.fingers.middle}, ring=${f.fingers.ring}, pinky=${f.fingers.pinky})`,
    );
  }
  const total = result.fingerStates.reduce((sum, f) => sum + f.count, 0);
  lines.push(`Total extended fingers visible: ${total}`);
  if (result.staticGestures?.length) {
    lines.push(`Static gestures: ${result.staticGestures.map((g) => g.name).join(", ")}`);
  }
  return lines.join("\n");
};
