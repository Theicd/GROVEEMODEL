/** Block premature stress/break packs during observation warmup. */

import type { DialogueContext } from "../vision2/types";
import type { SituationPack } from "./types";

export const BOOT_WARMUP_SEC = 18;

const BOOT_ALLOWED_IDS = new Set([
  "greeting-entry",
  "attention-seeking-wave",
  "direct-address",
  "acknowledgment-request",
  "social-bounce",
]);

const BLOCKED_DURING_BOOT = /stress|break|disengage|boredom|overload|patience|fatigue|drowsy/i;

export const isObservationWarmup = (dialogue: DialogueContext): boolean => {
  const sceneAge = dialogue.worldState.session.sceneAgeSec ?? 0;
  const stableSec = dialogue.consciousness?.stabilitySec ?? 0;
  if (sceneAge >= BOOT_WARMUP_SEC && stableSec >= 2) return false;
  return sceneAge < BOOT_WARMUP_SEC || stableSec < 1.5;
};

export const isPackBlockedDuringBoot = (pack: SituationPack, dialogue: DialogueContext): boolean => {
  if (!isObservationWarmup(dialogue)) return false;
  if (BOOT_ALLOWED_IDS.has(pack.id)) return false;
  if (pack.sceneTags?.includes("boot")) return false;
  if (pack.id.startsWith("entity-")) return false;
  if (BLOCKED_DURING_BOOT.test(pack.id)) return true;
  if (pack.sceneTags?.includes("stress") || pack.sceneTags?.includes("rest")) return true;
  if (pack.priority === "low" || pack.priority === "medium") return true;
  return false;
};
