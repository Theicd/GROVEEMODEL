/**
 * Continuous live context from vision-lab frames (updated every tick).
 * Separate from boot deep-vision summary — never replaces room baseline.
 */

import type { VisionResult } from "./vision-lab/core/types";
import type { WorldMemory } from "./worldMemory";

export const buildLiveContextFromResult = (result: VisionResult, world: WorldMemory): string => {
  const parts: string[] = [];

  if (world.personPresent) {
    parts.push(`Person present, posture: ${world.poseState}`);
  } else {
    parts.push("No person confirmed in frame");
  }

  if (world.fingerStates.length) {
    const fingerLine = world.fingerStates
      .map((f) => `${f.hand} hand: ${f.count} finger(s) extended`)
      .join("; ");
    parts.push(fingerLine);
  }

  if (result.staticGestures.length) {
    parts.push(
      `Hand signs: ${result.staticGestures.map((g) => `${g.hand} ${g.name}`).join(", ")}`,
    );
  }

  if (result.motionGestures.length) {
    parts.push(`Motion: ${result.motionGestures.map((g) => g.name).join(", ")}`);
  }

  if (result.bodyLanguage.length) {
    parts.push(
      result.bodyLanguage
        .slice(0, 4)
        .map((c) => `${c.signal} (${c.meaning})`)
        .join("; "),
    );
  }

  if (world.objects.length) {
    parts.push(`Objects in room: ${world.objects.slice(0, 8).join(", ")}`);
  }

  if (world.emotionEstimate) {
    parts.push(`Expression estimate: ${world.emotionEstimate}`);
  }

  if (result.sceneDescription?.trim()) {
    parts.push(result.sceneDescription.trim());
  }

  return parts.join(". ").slice(0, 900);
};
