import { buildPollinationsUrl, normalizePollinationsModel } from "../cloudImage";
import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../ui/useUiLanguage";

export type CloudImageGenResult =
  | { ok: true; url: string; promptUsed: string }
  | { ok: false; message: string };

/** Build Pollinations URL; translate Hebrew prompts to English for better results. */
export async function runCloudImageGeneration(
  prompt: string,
  pollinationsModel: string,
  uiLang: ChatUiLanguage,
): Promise<CloudImageGenResult> {
  const raw = prompt.trim();
  if (!raw) return { ok: false, message: "חסר תיאור לתמונה" };

  let promptUsed = raw;
  if (uiLang === "he" || /[\u0590-\u05FF]/.test(raw)) {
    try {
      const [en] = await translateTexts([raw], "en");
      if (en?.trim()) promptUsed = en.trim();
    } catch {
      /* use raw */
    }
  }

  try {
    const url = buildPollinationsUrl({
      prompt: promptUsed,
      model: normalizePollinationsModel(pollinationsModel),
    });
    return { ok: true, url, promptUsed };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, message: msg };
  }
}
