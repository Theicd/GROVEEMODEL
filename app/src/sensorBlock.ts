/** Structured sensor context for Gemma — adapted from JARVIS-VISION build_scene_sensor_block. */

import type { PersonPoseState } from "./poseHeuristics";
import type { WorldMemory } from "./worldMemory";

export type SensorSnapshot = {
  peopleCount: number;
  personPresent: boolean;
  poseState?: string;
  gestures?: string[];
  holding?: string[];
  focusHint?: string;
  motionLevel?: number;
  absentSec?: number;
};

export type ProactiveTrigger = {
  reason: string;
  topic: string;
  fallbackHint: string;
};

const SITUATION_SUBJECT =
  /^(wave|arm_movement|motion_burst|focused_work|pose_change|stood_with_drink)$|^object_held:/;

export const isSituationSubject = (subject: string): boolean =>
  SITUATION_SUBJECT.test(subject) || subject.startsWith("pose_change:");

export const poseFromWorld = (world: WorldMemory): PersonPoseState | null =>
  world.poseState !== "unknown" || world.poseConfidence > 0
    ? {
        poseState: world.poseState,
        confidence: world.poseConfidence,
        gestures: world.gestures,
        holding: world.holding,
        focusHint: world.focusHint,
      }
    : null;

export const describeSituationSubject = (subject: string, world: WorldMemory): string | null => {
  if (subject === "stood_with_drink") {
    const drink = world.holding.find((h) => /cup|bottle/.test(h)) ?? "drink";
    return `stood up while holding ${drink} — likely a break or transition, not a security report`;
  }
  if (subject.startsWith("object_held:")) {
    const item = subject.slice("object_held:".length);
    return `just picked up ${item} — curious about intent, not an object inventory`;
  }
  switch (subject) {
    case "wave":
      return "waving or gesturing toward camera — acknowledge attention, stay warm not robotic";
    case "arm_movement":
      return "arm movement in upper frame — maybe showing something or adjusting";
    case "motion_burst":
      return "sudden high motion — check in gently";
    case "focused_work":
      return "appears focused on something in front — respect concentration";
    case "pose_change":
      return "posture shifted (e.g. sitting to standing) — note the transition tentatively";
    default:
      return null;
  }
};

const formatLatestSituation = (world: WorldMemory): string | null => {
  if (!world.lastSituationSubject || !world.lastSituationAt) return null;
  const ageSec = Math.floor((Date.now() - world.lastSituationAt) / 1000);
  if (ageSec > 120) return null;
  const hint = describeSituationSubject(world.lastSituationSubject, world);
  if (!hint) return null;
  return ageSec <= 1 ? `Latest action: ${hint}` : `Latest action (${ageSec}s ago): ${hint}`;
};

export const buildSensorBlock = (world: WorldMemory, pose?: PersonPoseState | null): string => {
  const lines: string[] = [];

  if (world.personPresent) {
    const parts: string[] = ["present"];
    if (pose?.poseState && pose.poseState !== "unknown") parts.push(pose.poseState);
    if (pose?.gestures?.length) parts.push(`gestures: ${pose.gestures.join(", ")}`);
    if (pose?.holding?.length) parts.push(`holding: ${pose.holding.join(", ")}`);
    else if (world.holding.length) parts.push(`holding: ${world.holding.join(", ")}`);
    if (pose?.focusHint) parts.push(`focus: ${pose.focusHint}`);
    lines.push(`Person: ${parts.join("; ")}`);
  } else {
    const absentSec = world.absentSince ? Math.floor(world.msSinceAbsent() / 1000) : 0;
    lines.push(absentSec > 0 ? `Person: absent (${absentSec}s)` : "Person: not in frame");
  }

  if (world.lastMotionLevel > 0.01) {
    lines.push(`Motion level: ${world.lastMotionLevel.toFixed(2)}`);
  }

  const latestSituation = formatLatestSituation(world);
  if (latestSituation) lines.push(latestSituation);

  if (world.lastSummary.trim()) {
    lines.push(`Atmosphere: ${world.lastSummary.trim()}`);
  }

  const recent = world.lastChanges.slice(0, 3);
  if (recent.length) {
    lines.push(`Recent: ${recent.map((e) => e.text).join(" | ")}`);
  }

  return lines.length ? lines.join("\n") : "No sensor data yet.";
};

/** Richer block for Gemma proactive utterance — includes trigger + template hint. */
export const buildProactiveSensorBlock = (
  world: WorldMemory,
  pose?: PersonPoseState | null,
  trigger?: ProactiveTrigger,
): string => {
  const lines = [buildSensorBlock(world, pose)];

  if (trigger?.reason.startsWith("situation:")) {
    const sub = trigger.reason.slice("situation:".length);
    const hint = describeSituationSubject(sub, world);
    if (hint) lines.push(`Why speaking now: ${hint}`);
  }

  if (trigger?.fallbackHint.trim()) {
    lines.push(`Template intent (rephrase naturally, same meaning): ${trigger.fallbackHint.trim()}`);
  }

  return lines.join("\n");
};

export const buildSituationPrompt = (world: WorldMemory, pose?: PersonPoseState | null): string =>
  `Situation sensors (for your reasoning — do NOT read aloud as a list):\n${buildSensorBlock(world, pose)}`;
