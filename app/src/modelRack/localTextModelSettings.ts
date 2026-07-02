import { SMOLLM_CHAT_SYSTEM_EN_UI, smollmCoreSystemPrompt } from "./localTextTranslate";

export type LocalTextInferenceBackend = "auto" | "webgpu" | "wasm";

export type LocalTextModelSettings = {
  inferenceBackend: LocalTextInferenceBackend;
  temperature: number;
  maxNewTokens: number;
  maxNewTokensSearch: number;
  maxNewTokensGreeting: number;
  topP: number;
  /** Max chat messages sent to the model (first 2 pinned + recent tail). */
  historyTurns: number;
  /** Max characters of web-search context injected into the system prompt. */
  webBriefChars: number;
  /** Base system prompt (English). Hebrew UI uses translation bridge + English reply. */
  systemPrompt: string;
};

export const DEFAULT_LOCAL_TEXT_SETTINGS: LocalTextModelSettings = {
  inferenceBackend: "auto",
  temperature: 0.35,
  maxNewTokens: 192,
  maxNewTokensSearch: 280,
  maxNewTokensGreeting: 40,
  topP: 0.85,
  historyTurns: 12,
  webBriefChars: 420,
  systemPrompt: SMOLLM_CHAT_SYSTEM_EN_UI,
};

export function mergeLocalTextSettings(
  partial?: Partial<LocalTextModelSettings> | null,
): LocalTextModelSettings {
  const merged = { ...DEFAULT_LOCAL_TEXT_SETTINGS, ...partial };
  const systemPrompt =
    typeof merged.systemPrompt === "string" && merged.systemPrompt.trim()
      ? merged.systemPrompt.trim()
      : DEFAULT_LOCAL_TEXT_SETTINGS.systemPrompt;
  const historyTurnsRaw =
    merged.historyTurns === 6 ? DEFAULT_LOCAL_TEXT_SETTINGS.historyTurns : merged.historyTurns;
  return {
    ...merged,
    systemPrompt,
    inferenceBackend:
      merged.inferenceBackend === "wasm" ||
      merged.inferenceBackend === "webgpu" ||
      merged.inferenceBackend === "auto"
        ? merged.inferenceBackend
        : "auto",
    historyTurns: clampInt(historyTurnsRaw, 2, 48, DEFAULT_LOCAL_TEXT_SETTINGS.historyTurns),
    webBriefChars: clampInt(merged.webBriefChars, 200, 2000, DEFAULT_LOCAL_TEXT_SETTINGS.webBriefChars),
    maxNewTokens: clampInt(merged.maxNewTokens, 32, 1024, DEFAULT_LOCAL_TEXT_SETTINGS.maxNewTokens),
    maxNewTokensSearch: clampInt(
      merged.maxNewTokensSearch,
      64,
      1024,
      DEFAULT_LOCAL_TEXT_SETTINGS.maxNewTokensSearch,
    ),
    maxNewTokensGreeting: clampInt(
      merged.maxNewTokensGreeting,
      16,
      256,
      DEFAULT_LOCAL_TEXT_SETTINGS.maxNewTokensGreeting,
    ),
  };
}

/** System prompt base for the active UI language before search/game append blocks. */
export function localTextBaseSystemForUi(
  uiLang: "he" | "en",
  settings: LocalTextModelSettings,
): string {
  const custom = settings.systemPrompt.trim() || DEFAULT_LOCAL_TEXT_SETTINGS.systemPrompt;
  return smollmCoreSystemPrompt(uiLang === "en" ? custom : undefined);
}

function clampInt(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}
