import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../ui/useUiLanguage";

export type LocalTextChatTurn = { role: "user" | "assistant"; content: string };

export type LocalTextHistoryEntry = {
  role: "user" | "assistant";
  content: string;
  /** English text sent to SmolLM for this assistant turn (Hebrew bridge). */
  modelDraft?: string;
  /** English text sent to SmolLM for this user turn (Hebrew bridge). */
  userModelDraft?: string;
};

/** Minimal identity — SmolLM collapses on long system blocks. */
export const SMOLLM_GROVEE_IDENTITY =
  "You are Groovie (גרובי), a friendly chat assistant in GROVEEMODEL. Reply in 1-2 short natural sentences like a simple chat bot.";

/** SmolLM works best in English — used when UI is Hebrew and we bridge via translate. */
export const SMOLLM_MODEL_SYSTEM_EN =
  "You are a concise, factual assistant. Answer in 1-3 short sentences using only facts you are sure of. If unsure, say you don't know. Do not repeat yourself and stop when done. For math, give only the numeric result.";

export const SMOLLM_CHAT_SYSTEM_EN_UI =
  "You are a concise, factual assistant. Answer in 1-3 short sentences in the user's language. Use only facts you are sure of; if unsure, say you don't know. Do not repeat yourself and stop when done.";

/** SmolLM2 practical context — used for context ring display. */
export const LOCAL_TEXT_CONTEXT_BUDGET_CHARS = 4096 * 4;

export const LOCAL_TEXT_HISTORY_CHAR_BUDGET = 3000;

/** Pin user "remember …" messages (not generic opening chitchat). */
export const LOCAL_TEXT_HISTORY_PIN_HEAD = 0;

export function localTextHistoryEntryCharCost(e: LocalTextHistoryEntry): number {
  const draft =
    e.role === "assistant"
      ? e.modelDraft?.trim() || e.content
      : e.userModelDraft?.trim() || e.content;
  return draft.length + 64;
}

/**
 * Select history for SmolLM: pin first messages + recent tail, capped by slot count and char budget.
 * Returns sourceIndices so draft backfill maps to the correct chat rows.
 */
export function buildLocalTextHistoryForModel(
  allEntries: LocalTextHistoryEntry[],
  options: {
    maxMessageSlots?: number;
    maxChars?: number;
    pinHead?: number;
    /** Always include these message indices (e.g. "remember Paris" mid-chat). */
    pinnedSourceIndices?: number[];
  } = {},
): { entries: LocalTextHistoryEntry[]; sourceIndices: number[] } {
  const maxMessageSlots = options.maxMessageSlots ?? 12;
  const maxChars = options.maxChars ?? LOCAL_TEXT_HISTORY_CHAR_BUDGET;
  const pinHead = options.pinHead ?? LOCAL_TEXT_HISTORY_PIN_HEAD;
  const pinnedSet = new Set(options.pinnedSourceIndices ?? []);

  const rows = allEntries.map((entry, sourceIndex) => ({
    entry,
    sourceIndex,
    cost: localTextHistoryEntryCharCost(entry),
    pinned: pinnedSet.has(sourceIndex),
  }));
  if (!rows.length) return { entries: [], sourceIndices: [] };

  const pinnedRows = rows.filter((r) => r.pinned);
  const unpinnedRows = rows.filter((r) => !r.pinned);

  let headRows = pinHead > 0 ? unpinnedRows.slice(0, pinHead) : [];
  const reserved = new Set([...pinnedRows, ...headRows].map((r) => r.sourceIndex));
  const tailBudget = Math.max(0, maxMessageSlots - pinnedRows.length - headRows.length);
  const tailRows: typeof rows = [];
  for (let i = unpinnedRows.length - 1; i >= headRows.length && tailRows.length < tailBudget; i--) {
    const row = unpinnedRows[i];
    if (reserved.has(row.sourceIndex)) continue;
    tailRows.unshift(row);
    reserved.add(row.sourceIndex);
  }

  if (pinnedRows.length + headRows.length + tailRows.length > maxMessageSlots) {
    headRows = [];
    tailRows.length = 0;
    for (let i = unpinnedRows.length - 1; i >= 0 && tailRows.length < maxMessageSlots - pinnedRows.length; i--) {
      const row = unpinnedRows[i];
      if (pinnedSet.has(row.sourceIndex)) continue;
      tailRows.unshift(row);
    }
  }

  let windowed = [...pinnedRows, ...headRows, ...tailRows].sort(
    (a, b) => a.sourceIndex - b.sourceIndex,
  );

  const mustKeep = windowed.filter((r) => r.pinned);
  const mustKeepCost = mustKeep.reduce((sum, row) => sum + row.cost, 0);
  let budget = Math.max(0, maxChars - mustKeepCost);
  const optional = windowed.filter((r) => !r.pinned);
  const keptOptional: typeof rows = [];
  for (let i = optional.length - 1; i >= 0; i--) {
    const row = optional[i];
    if (row.cost <= budget) {
      keptOptional.unshift(row);
      budget -= row.cost;
    } else if (!keptOptional.length) {
      break;
    } else {
      break;
    }
  }

  const final = [...mustKeep, ...keptOptional].sort((a, b) => a.sourceIndex - b.sourceIndex);
  return {
    entries: final.map((row) => row.entry),
    sourceIndices: final.map((row) => row.sourceIndex),
  };
}

/** @deprecated Prefer buildLocalTextHistoryForModel */
export function trimLocalTextHistory(
  entries: LocalTextHistoryEntry[],
  maxChars = LOCAL_TEXT_HISTORY_CHAR_BUDGET,
): LocalTextHistoryEntry[] {
  return buildLocalTextHistoryForModel(entries, { maxChars, maxMessageSlots: entries.length }).entries;
}

export function smollmCoreSystemPrompt(customEnUi?: string): string {
  const tail =
    customEnUi?.trim() && customEnUi.trim() !== SMOLLM_CHAT_SYSTEM_EN_UI
      ? customEnUi.trim()
      : "Answer in 1-3 short sentences using only facts you are sure of.";
  return `${SMOLLM_GROVEE_IDENTITY}\n${tail}`;
}

export function needsLocalTextTranslationBridge(uiLang: ChatUiLanguage): boolean {
  return uiLang === "he";
}

export function smollmSystemPromptForUi(uiLang: ChatUiLanguage): string {
  return smollmCoreSystemPrompt();
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
  history: LocalTextHistoryEntry[],
  uiLang: ChatUiLanguage,
  baseSystemPrompt?: string,
): Promise<{
  prompt: string;
  history: LocalTextChatTurn[];
  systemPrompt: string;
  userModelDraft?: string;
  historyDraftUpdates: Array<{ index: number; userModelDraft?: string; modelDraft?: string }>;
}> {
  const systemForUi = baseSystemPrompt?.trim() || smollmSystemPromptForUi(uiLang);
  const historyDraftUpdates: Array<{ index: number; userModelDraft?: string; modelDraft?: string }> =
    [];

  if (!needsLocalTextTranslationBridge(uiLang)) {
    return {
      prompt,
      history: history.map((h) => ({ role: h.role, content: h.content })),
      systemPrompt: systemForUi,
      historyDraftUpdates,
    };
  }

  const modelPrompt = await localTextToModelLanguage(prompt, uiLang);
  const modelHistory: LocalTextChatTurn[] = [];
  for (let i = 0; i < history.length; i++) {
    const entry = history[i];
    const saved =
      entry.role === "assistant"
        ? entry.modelDraft?.trim()
        : entry.userModelDraft?.trim();
    if (saved) {
      modelHistory.push({ role: entry.role, content: saved });
      continue;
    }
    const content = await localTextToModelLanguage(entry.content, uiLang);
    modelHistory.push({ role: entry.role, content });
    if (entry.role === "assistant") {
      historyDraftUpdates.push({ index: i, modelDraft: content });
    } else {
      historyDraftUpdates.push({ index: i, userModelDraft: content });
    }
  }

  return {
    prompt: modelPrompt,
    history: modelHistory,
    systemPrompt: systemForUi,
    userModelDraft: modelPrompt,
    historyDraftUpdates,
  };
}
