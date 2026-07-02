import {
  DEFAULT_LOCAL_TEXT_SETTINGS,
  mergeLocalTextSettings,
  type LocalTextModelSettings,
} from "./localTextModelSettings";
import {
  HUNYUAN_HF_MODEL_ID,
  HUNYUAN_RACK_ID,
  SMOLLM_135M_HF_MODEL_ID,
  SMOLLM_135M_RACK_ID,
  SMOLLM_HF_MODEL_ID,
  SMOLLM_RACK_ID,
} from "./localTextModels";

/** Per-model inference tuning passed to the text worker. */
export type LocalTextGenerationProfile = {
  temperatureDefault: number;
  repetitionPenalty: number;
  repetitionPenaltyWasm: number;
  topPDefault: number;
  topK: number;
  topKWasm: number;
  wasmMaxNewTokensCap: number;
  noRepeatNgramSize: number;
};

export type LocalTextModelProfile = {
  rackId: string;
  hfModelId: string;
  label: string;
  hintHe: string;
  estimatedBytes: number;
  settingsOverrides: Partial<LocalTextModelSettings>;
  historyCharBudget: number;
  contextBudgetChars: number;
  systemMaxChars: number;
  generation: LocalTextGenerationProfile;
};

const SMOLLM_GENERATION: LocalTextGenerationProfile = {
  temperatureDefault: 0.35,
  repetitionPenalty: 1.15,
  repetitionPenaltyWasm: 1.3,
  topPDefault: 0.85,
  topK: 50,
  topKWasm: 40,
  wasmMaxNewTokensCap: 160,
  noRepeatNgramSize: 3,
};

const HUNYUAN_GENERATION: LocalTextGenerationProfile = {
  temperatureDefault: 0.7,
  repetitionPenalty: 1.1,
  repetitionPenaltyWasm: 1.15,
  topPDefault: 0.9,
  topK: 50,
  topKWasm: 45,
  wasmMaxNewTokensCap: 320,
  noRepeatNgramSize: 3,
};

const SMOLLM_PROFILE: LocalTextModelProfile = {
  rackId: SMOLLM_RACK_ID,
  hfModelId: SMOLLM_HF_MODEL_ID,
  label: "SmolLM2 360M",
  hintHe: "מודל מקומי · שיחה טקסט",
  estimatedBytes: 220_000_000,
  settingsOverrides: {},
  historyCharBudget: 3000,
  contextBudgetChars: 4096 * 4,
  systemMaxChars: 900,
  generation: SMOLLM_GENERATION,
};

const SMOLLM_135M_PROFILE: LocalTextModelProfile = {
  ...SMOLLM_PROFILE,
  rackId: SMOLLM_135M_RACK_ID,
  hfModelId: SMOLLM_135M_HF_MODEL_ID,
  label: "SmolLM2 135M",
  estimatedBytes: 175_000_000,
};

/** Hunyuan 0.5B Instruct — long-context chat (ONNX q4 ~380MB). */
const HUNYUAN_PROFILE: LocalTextModelProfile = {
  rackId: HUNYUAN_RACK_ID,
  hfModelId: HUNYUAN_HF_MODEL_ID,
  label: "Hunyuan 0.5B",
  hintHe: "הקשר ארוך · זיכרון שיחה",
  estimatedBytes: 480_000_000,
  settingsOverrides: {
    temperature: 0.7,
    maxNewTokens: 384,
    maxNewTokensSearch: 480,
    maxNewTokensGreeting: 48,
    topP: 0.9,
    historyTurns: 40,
    webBriefChars: 560,
  },
  historyCharBudget: 64_000,
  contextBudgetChars: 8192 * 4,
  systemMaxChars: 1800,
  generation: HUNYUAN_GENERATION,
};

const PROFILES: LocalTextModelProfile[] = [SMOLLM_PROFILE, SMOLLM_135M_PROFILE, HUNYUAN_PROFILE];

const BY_HF = new Map(PROFILES.map((p) => [p.hfModelId, p] as const));
const BY_RACK = new Map(PROFILES.map((p) => [p.rackId, p] as const));

export function localTextProfileForHfModelId(hfModelId: string): LocalTextModelProfile {
  return BY_HF.get(hfModelId) ?? SMOLLM_PROFILE;
}

export function localTextProfileForRackId(rackId: string): LocalTextModelProfile | null {
  return BY_RACK.get(rackId) ?? null;
}

export function localTextSettingsForProfile(
  profile: LocalTextModelProfile,
  base?: Partial<LocalTextModelSettings> | null,
): LocalTextModelSettings {
  return mergeLocalTextSettings({ ...DEFAULT_LOCAL_TEXT_SETTINGS, ...base, ...profile.settingsOverrides });
}

export function isHunyuanLocalTextModel(hfModelId: string): boolean {
  return hfModelId === HUNYUAN_HF_MODEL_ID;
}

/** Phone-safe history cap — Hunyuan supports long context; limit by device RAM. */
export function resolveHistoryCharBudget(profile: LocalTextModelProfile): number {
  const base = profile.historyCharBudget;
  if (profile.rackId !== HUNYUAN_RACK_ID) return base;
  if (typeof navigator === "undefined") return base;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (typeof mem === "number" && mem <= 4) return Math.min(base, 24_000);
  if (typeof mem === "number" && mem <= 6) return Math.min(base, 40_000);
  return base;
}

export const HUNYUAN_ESTIMATED_BYTES = HUNYUAN_PROFILE.estimatedBytes;
