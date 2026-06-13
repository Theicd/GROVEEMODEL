export type ChatHardwareProfileId = "ultra" | "balanced" | "safe" | "low";

export type ChatProfileBudgets = {
  id: ChatHardwareProfileId;
  labelHe: string;
  totalPromptChars: number;
  historyChars: number;
  webBriefChars: number;
  maxNewTokensDefault: number;
  maxNewTokensSearch: number;
  maxNewTokensCode: number;
};

const PROFILES: Record<ChatHardwareProfileId, ChatProfileBudgets> = {
  ultra: {
    id: "ultra",
    labelHe: "Ultra (32GB+)",
    totalPromptChars: 22_000,
    historyChars: 14_000,
    webBriefChars: 800,
    maxNewTokensDefault: 768,
    maxNewTokensSearch: 512,
    maxNewTokensCode: 1536,
  },
  balanced: {
    id: "balanced",
    labelHe: "Balanced (16GB)",
    totalPromptChars: 18_000,
    historyChars: 12_000,
    webBriefChars: 700,
    maxNewTokensDefault: 512,
    maxNewTokensSearch: 384,
    maxNewTokensCode: 1024,
  },
  safe: {
    id: "safe",
    labelHe: "Safe",
    totalPromptChars: 14_000,
    historyChars: 10_000,
    webBriefChars: 600,
    maxNewTokensDefault: 384,
    maxNewTokensSearch: 320,
    maxNewTokensCode: 768,
  },
  low: {
    id: "low",
    labelHe: "Low RAM",
    totalPromptChars: 10_000,
    historyChars: 6_000,
    webBriefChars: 500,
    maxNewTokensDefault: 256,
    maxNewTokensSearch: 256,
    maxNewTokensCode: 512,
  },
};

const STORAGE_KEY = "grovee_chat_profile_v1";

export const getProfileBudgets = (id: ChatHardwareProfileId): ChatProfileBudgets => PROFILES[id];

export const listChatProfiles = (): ChatProfileBudgets[] => Object.values(PROFILES);

export const detectChatHardwareProfile = (): ChatHardwareProfileId => {
  const stored = loadChatProfileOverride();
  if (stored) return stored;
  const mem = typeof navigator !== "undefined" ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory : undefined;
  const cores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency : undefined;
  if (mem != null && mem >= 8 && (cores ?? 4) >= 8) return "ultra";
  if (mem != null && mem >= 4) return "balanced";
  if (mem != null && mem <= 2) return "low";
  return "safe";
};

export const loadChatProfileOverride = (): ChatHardwareProfileId | null => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw && raw in PROFILES) return raw as ChatHardwareProfileId;
  } catch {
    /* ignore */
  }
  return null;
};

export const saveChatProfileOverride = (id: ChatHardwareProfileId): void => {
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    /* ignore */
  }
};
