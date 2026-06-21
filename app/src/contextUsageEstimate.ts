import type { ContextUsage } from "./ContextRing";
import type { ChatTurn } from "./chatIntents";
import { prepareChatContext, type PreparedChatContext } from "./chatResourceBudget";
import {
  detectChatHardwareProfile,
  getProfileBudgets,
  type ChatHardwareProfileId,
} from "./chatHardwareProfile";

export const CHARS_PER_TOKEN = 4;

export const approxTokensFromChars = (chars: number): number =>
  Math.max(0, Math.round(chars / CHARS_PER_TOKEN));

export const formatTokenCount = (tokens: number): string => {
  if (tokens >= 10_000) return `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
  return tokens.toLocaleString("he-IL");
};

export const toContextUsage = (
  prepared: PreparedChatContext,
  profileLabel: string,
): ContextUsage => ({
  percent: prepared.staminaPercent,
  usedChars: prepared.usedChars,
  totalBudget: prepared.totalBudget,
  profileLabel,
  breakdown: prepared.breakdown,
});

export type LiveContextEstimateInput = {
  history: ChatTurn[];
  draftPrompt: string;
  systemPromptChars: number;
  webContextChars?: number;
  imageCount: number;
  profileId?: ChatHardwareProfileId;
  isSearchTurn?: boolean;
  isCodeTurn?: boolean;
};

/** Live estimate for the context ring — mirrors prepareChatContext sizing. */
export const estimateLiveContextUsage = (input: LiveContextEstimateInput): ContextUsage => {
  const profileId = input.profileId ?? detectChatHardwareProfile();
  const systemPrompt = input.systemPromptChars > 0 ? " ".repeat(input.systemPromptChars) : "";
  const webContext = input.webContextChars ? " ".repeat(input.webContextChars) : "";
  const prepared = prepareChatContext({
    history: input.history,
    webContext,
    systemPrompt,
    userPrompt: input.draftPrompt.trim(),
    imageCount: input.imageCount ?? 0,
    maxNewTokens: getProfileBudgets(profileId).maxNewTokensDefault,
    profileId,
    isSearchTurn: input.isSearchTurn,
    isCodeTurn: input.isCodeTurn,
  });
  return toContextUsage(prepared, getProfileBudgets(profileId).labelHe);
};

/** Default system prompt size before first measured turn. */
export const defaultSystemPromptChars = (baseSystemPrompt: string, cameraMode: boolean): number =>
  baseSystemPrompt.length + (cameraMode ? 2800 : 900);
