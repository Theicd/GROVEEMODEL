import type { UiLang } from "../chatRoutePrelude";
import {
  isImageGenerateRequest,
  resolveImagePromptFromHistory,
  type ChatTurnLike,
  type GeneratedImagePayload,
} from "./imageIntent";
import { runCloudImageGeneration } from "./runCloudImageGen";

export type InChatImageGenDeps = {
  trimmed: string;
  priorTurns: ChatTurnLike[];
  uiLang: UiLang;
  preferredPollinationsModel: string;
  pendingImagePromptRef: { current: string | null };
  pendingGeneratedImageRef: { current: GeneratedImagePayload | null };
  setStatus: (s: string) => void;
  deliverCanned: (
    reply: string,
    webContext: string,
    replySource: string,
    activityTitle?: string,
  ) => void;
};

/** Returns true when the turn was fully handled (image generated or error canned). */
export async function tryInChatImageGeneration(deps: InChatImageGenDeps): Promise<boolean> {
  if (!isImageGenerateRequest(deps.trimmed)) return false;

  const imgPrompt = resolveImagePromptFromHistory(
    deps.trimmed,
    deps.pendingImagePromptRef.current,
    deps.priorTurns,
  );
  if (!imgPrompt) {
    deps.deliverCanned(
      deps.uiLang === "he"
        ? "אין עדיין תיאור לתמונה. כתוב קודם «תאר לי …» ואז «צור מזה תמונה»."
        : "No description yet. Ask me to describe something first, then «create an image from that».",
      "",
      "image-gen",
      deps.uiLang === "he" ? "יצירת תמונה" : "Image generation",
    );
    return true;
  }

  deps.setStatus(deps.uiLang === "he" ? "מייצר תמונה…" : "Generating image…");
  const gen = await runCloudImageGeneration(
    imgPrompt,
    deps.preferredPollinationsModel,
    deps.uiLang,
  );
  if (!gen.ok) {
    deps.deliverCanned(
      deps.uiLang === "he"
        ? `⚠️ לא הצלחתי ליצור תמונה: ${gen.message}`
        : `⚠️ Image failed: ${gen.message}`,
      "",
      "image-gen",
    );
    return true;
  }

  deps.pendingGeneratedImageRef.current = { url: gen.url, prompt: gen.promptUsed };
  deps.pendingImagePromptRef.current = gen.promptUsed;
  deps.deliverCanned(
    deps.uiLang === "he" ? "הנה התמונה לפי התיאור." : "Here is the image from your description.",
    "",
    "image-gen",
    "Pollinations · תמונה",
  );
  return true;
}
