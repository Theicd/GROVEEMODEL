/** Fresh person-state snapshot for chat queries (Focus On Person). */

import type { PoseSource } from "./poseDetector";

export type PersonFocusSnapshot = {
  personPresent: boolean;
  poseState: "standing" | "sitting" | "unknown";
  confidence: number;
  holding: string[];
  gestures: string[];
  focusHint: string;
  poseSource: PoseSource;
  capturedAt: number;
  validationFrames: number;
};

export const formatFreshPersonBlock = (focus: PersonFocusSnapshot): string => {
  const lines = [
    "FRESH PERSON ANALYSIS (authoritative — answer from this + attached snapshot, NOT older memory):",
    `Captured: just now (${focus.validationFrames} validation frame(s))`,
    `Person in frame: ${focus.personPresent ? "yes" : "no"}`,
  ];

  if (!focus.personPresent) {
    lines.push("Do NOT claim sitting/standing — no person detected in fresh pass.");
    return lines.join("\n");
  }

  if (focus.confidence < 0.45 || focus.poseState === "unknown") {
    lines.push(
      `Posture: uncertain (confidence ${focus.confidence.toFixed(2)}, source ${focus.poseSource})`,
    );
    lines.push('Say tentatively — e.g. "קשה לדעת בוודאות" / "נראה לי…" — do NOT assert sitting or standing as fact.');
  } else {
    lines.push(
      `Posture: ${focus.poseState} (confidence ${focus.confidence.toFixed(2)}, source ${focus.poseSource})`,
    );
  }

  lines.push(
    focus.holding.length ? `Holding: ${focus.holding.join(", ")}` : "Holding: none detected",
  );
  if (focus.gestures.length) lines.push(`Gestures: ${focus.gestures.join(", ")}`);
  if (focus.focusHint) lines.push(`Focus: ${focus.focusHint}`);

  return lines.join("\n");
};
