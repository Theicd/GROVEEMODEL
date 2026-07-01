import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../ui/useUiLanguage";

export type LocalTextChatTurn = { role: "user" | "assistant"; content: string };

/** SmolLM works best in English — used when UI is Hebrew and we bridge via translate. */
export const SMOLLM_MODEL_SYSTEM_EN =
  "You are a concise, factual assistant. Answer in 1-3 short sentences using only facts you are sure of. If unsure, say you don't know. Do not repeat yourself and stop when done. For math, give only the numeric result.";

export const SMOLLM_CHAT_SYSTEM_EN_UI =
  "You are a concise, factual assistant. Answer in 1-3 short sentences in the user's language. Use only facts you are sure of; if unsure, say you don't know. Do not repeat yourself and stop when done.";

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
    const out = await translateOne(text, "en", "he");
    console.info(`[translate] he→en ok: "${text.slice(0, 40)}" → "${out.slice(0, 40)}"`);
    return out;
  } catch (err) {
    console.warn(
      `[translate] he→en FAILED, sending original Hebrew to model: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
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
    const out = await translateOne(text, "he", "en");
    console.info(`[translate] en→he ok: "${text.slice(0, 40)}" → "${out.slice(0, 40)}"`);
    return out;
  } catch (err) {
    console.warn(
      `[translate] en→he FAILED, showing English reply to user: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
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
