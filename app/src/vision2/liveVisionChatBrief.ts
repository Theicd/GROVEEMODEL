/**
 * Factual live vision brief for chat — bridges camera sensors → Gemma.
 */

import type { VisionResult } from "../vision-lab/core/types";
import type { WorldMemory } from "../worldMemory";
import type { DialogueContext } from "./types";
import { buildPreChatVisionReport } from "./preChatVisionReport";

export const CHAT_VISION_SYSTEM_HINT = `You have LIVE CAMERA data in [INTERNAL VISION CONTEXT] below.
Use personVisible, holding, emotion, faceData, posture when the user asks about sight/people/body.
If the user is chatting (game, boredom, story, feelings) — answer the conversation; do NOT lead with scene description.
If personVisible=yes — NEVER say "no person".
NEVER copy or echo the [INTERNAL VISION CONTEXT] block — user must NOT see it.
Answer in Hebrew, 1–3 sentences for dialogue; concrete facts only when asked.`;

export const buildLiveVisionChatBrief = (params: {
  vision: VisionResult | null;
  dialogue: DialogueContext | null;
  world: WorldMemory;
  cameraActive: boolean;
  snapshotAttached?: boolean;
}): string => buildPreChatVisionReport(params).text;

export const buildInternalVisionContextForUi = (params: {
  vision: VisionResult | null;
  dialogue: DialogueContext | null;
  world: WorldMemory;
  cameraActive: boolean;
  snapshotAttached?: boolean;
}): string => buildPreChatVisionReport(params).internalEn;

export const buildChatVisionContextBlock = (
  dialogue: DialogueContext | null,
  vision: VisionResult | null,
  world: WorldMemory,
  cameraActive: boolean,
  snapshotAttached = false,
): string =>
  buildLiveVisionChatBrief({
    vision,
    dialogue,
    world,
    cameraActive,
    snapshotAttached,
  });

export {
  buildPreChatVisionReport,
  buildInternalVisionContextEn,
  resolvePersonVisibleForChat,
} from "./preChatVisionReport";
