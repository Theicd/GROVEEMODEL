import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../ui/useUiLanguage";

export type LocalTextChatTurn = { role: "user" | "assistant"; content: string };

/** SmolLM works best in English — used when UI is Hebrew and we bridge via translate. */
export const SMOLLM_MODEL_SYSTEM_EN =
  "You are SmolLM, a helpful assistant. Reply clearly and concisely in English only.";

export const SMOLLM_CHAT_SYSTEM_EN_UI =
  "You are SmolLM, a helpful assistant. Reply clearly and concisely in the same language the user writes.";

export function needsLocalTextTranslationBridge(uiLang: ChatUiLanguage): boolean {
  return uiLang === "he";
}

export function smollmSystemPromptForUi(uiLang: ChatUiLanguage): string {
  return needsLocalTextTranslationBridge(uiLang)
    ? SMOLLM_MODEL_SYSTEM_EN
    : SMOLLM_CHAT_SYSTEM_EN_UI;
}

async function translateOne(text: string, target: string, source: string): Promise<string> {
  const trimmed = text.trim();
  if (!trimmed) return text;
  const { texts } = await translateTexts([trimmed], target, source);
  return texts[0]?.trim() || text;
}

/** Hebrew UI → English for the local text model. */
export async function localTextToModelLanguage(
  text: string,
  uiLang: ChatUiLanguage,
): Promise<string> {
  if (!needsLocalTextTranslationBridge(uiLang)) return text;
  try {
    return await translateOne(text, "en", "he");
  } catch {
    return text;
  }
}

/** Model English reply → Hebrew for Hebrew UI. */
export async function localTextToUiLanguage(
  text: string,
  uiLang: ChatUiLanguage,
): Promise<string> {
  if (!needsLocalTextTranslationBridge(uiLang)) return text;
  try {
    return await translateOne(text, "he", "en");
  } catch {
    return text;
  }
}

export async function translateLocalTextHistoryForModel(
  history: LocalTextChatTurn[],
  uiLang: ChatUiLanguage,
): Promise<LocalTextChatTurn[]> {
  if (!needsLocalTextTranslationBridge(uiLang) || !history.length) return history;

  const contents = history.map((t) => t.content);
  try {
    const { texts } = await translateTexts(contents, "en", "he");
    return history.map((turn, i) => ({
      role: turn.role,
      content: texts[i]?.trim() || turn.content,
    }));
  } catch {
    return history;
  }
}

export async function prepareLocalTextTurnForModel(
  prompt: string,
  history: LocalTextChatTurn[],
  uiLang: ChatUiLanguage,
  baseSystemPrompt?: string,
): Promise<{ prompt: string; history: LocalTextChatTurn[]; systemPrompt: string }> {
  const systemForUi = baseSystemPrompt?.trim() || smollmSystemPromptForUi(uiLang);

  if (!needsLocalTextTranslationBridge(uiLang)) {
    return {
      prompt,
      history,
      systemPrompt: systemForUi,
    };
  }

  const [modelPrompt, modelHistory] = await Promise.all([
    localTextToModelLanguage(prompt, uiLang),
    translateLocalTextHistoryForModel(history, uiLang),
  ]);

  return {
    prompt: modelPrompt,
    history: modelHistory,
    systemPrompt: systemForUi,
  };
}
