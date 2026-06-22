import {
  SMOLLM_CHAT_SYSTEM_EN_UI,
  SMOLLM_MODEL_SYSTEM_EN,
} from "./localTextTranslate";

export type LocalTextInferenceBackend = "auto" | "webgpu" | "wasm";

export type LocalTextModelSettings = {
  inferenceBackend: LocalTextInferenceBackend;
  temperature: number;
  maxNewTokens: number;
  maxNewTokensSearch: number;
  maxNewTokensGreeting: number;
  topP: number;
  /** How many prior chat turns to send to the model. */
  historyTurns: number;
  /** Max characters of web-search context injected into the system prompt. */
  webBriefChars: number;
  /** Base system prompt (English). Hebrew UI uses translation bridge + English reply. */
  systemPrompt: string;
};

export const DEFAULT_LOCAL_TEXT_SETTINGS: LocalTextModelSettings = {
  inferenceBackend: "auto",
  temperature: 0.7,
  maxNewTokens: 256,
  maxNewTokensSearch: 384,
  maxNewTokensGreeting: 48,
  topP: 0.9,
  historyTurns: 8,
  webBriefChars: 600,
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
  return {
    ...merged,
    systemPrompt,
    inferenceBackend:
      merged.inferenceBackend === "wasm" ||
      merged.inferenceBackend === "webgpu" ||
      merged.inferenceBackend === "auto"
        ? merged.inferenceBackend
        : "auto",
    historyTurns: clampInt(merged.historyTurns, 2, 24, DEFAULT_LOCAL_TEXT_SETTINGS.historyTurns),
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
  if (uiLang === "he") {
    if (/english only/i.test(custom)) return custom;
    return `${SMOLLM_MODEL_SYSTEM_EN}\n\n${custom}`;
  }
  return custom;
}

function clampInt(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.round(value)));
}
