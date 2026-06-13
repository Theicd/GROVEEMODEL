import type { ChatTurn } from "./chatIntents";
import { CHAT_HISTORY_CHAR_BUDGET, trimHistoryForContext } from "./chatIntents";
import type { ChatHardwareProfileId } from "./chatHardwareProfile";
import { detectChatHardwareProfile, getProfileBudgets } from "./chatHardwareProfile";

export const WEB_BRIEF_CHAR_BUDGET = 800;

export type PromptBudgetInput = {
  history: ChatTurn[];
  webContext: string;
  systemPrompt: string;
  userPrompt: string;
  imageCount: number;
  maxNewTokens: number;
  profileId?: ChatHardwareProfileId;
  pinLastAssistant?: boolean;
  isSearchTurn?: boolean;
  isCodeTurn?: boolean;
};

export type PreparedChatContext = {
  history: ChatTurn[];
  webContext: string;
  maxNewTokens: number;
  usedChars: number;
  totalBudget: number;
  staminaPercent: number;
  breakdown: { history: number; web: number; system: number; user: number; images: number };
};

const imageCharCost = (count: number) => count * 4000;

export const estimatePromptChars = (input: Omit<PromptBudgetInput, "profileId" | "pinLastAssistant">): number => {
  const historyChars = input.history.reduce((sum, t) => sum + t.content.length + 64, 0);
  return (
    historyChars +
    input.webContext.length +
    input.systemPrompt.length +
    input.userPrompt.length +
    imageCharCost(input.imageCount)
  );
};

export const resolveDynamicMaxNewTokens = (
  base: number,
  opts: { isSearchTurn?: boolean; isCodeTurn?: boolean; profileId?: ChatHardwareProfileId },
): number => {
  const profile = getProfileBudgets(opts.profileId ?? detectChatHardwareProfile());
  if (opts.isCodeTurn) return Math.min(base, profile.maxNewTokensCode);
  if (opts.isSearchTurn) return Math.min(base, profile.maxNewTokensSearch);
  return Math.min(base, profile.maxNewTokensDefault);
};

export const capWebContext = (webContext: string, maxChars = WEB_BRIEF_CHAR_BUDGET): string => {
  const t = webContext.trim();
  if (t.length <= maxChars) return t;
  return `${t.slice(0, maxChars - 20).trim()}\n…[truncated]`;
};

export const prepareChatContext = (input: PromptBudgetInput): PreparedChatContext => {
  const profileId = input.profileId ?? detectChatHardwareProfile();
  const budgets = getProfileBudgets(profileId);
  let webContext = capWebContext(input.webContext, budgets.webBriefChars);
  let maxNewTokens = resolveDynamicMaxNewTokens(input.maxNewTokens, {
    isSearchTurn: input.isSearchTurn,
    isCodeTurn: input.isCodeTurn,
    profileId,
  });

  let history = trimHistoryForContext(
    input.history,
    Math.min(budgets.historyChars, CHAT_HISTORY_CHAR_BUDGET),
    input.pinLastAssistant ?? false,
  );

  if (input.history.length > 10) {
    history = trimHistoryForContext(history.slice(-8), Math.max(1500, Math.floor(budgets.historyChars * 0.45)), false);
  }

  const breakdown = () => ({
    history: history.reduce((s, t) => s + t.content.length + 64, 0),
    web: webContext.length,
    system: input.systemPrompt.length,
    user: input.userPrompt.length,
    images: imageCharCost(input.imageCount),
  });

  let used = Object.values(breakdown()).reduce((a, b) => a + b, 0);
  const totalBudget = budgets.totalPromptChars;

  if (used > totalBudget) {
    history = trimHistoryForContext(history, Math.max(2000, budgets.historyChars - (used - totalBudget)), false);
    used = Object.values(breakdown()).reduce((a, b) => a + b, 0);
  }
  if (used > totalBudget && webContext.length > 400) {
    webContext = capWebContext(webContext, Math.max(400, webContext.length - (used - totalBudget)));
    used = Object.values(breakdown()).reduce((a, b) => a + b, 0);
  }
  if (used > totalBudget) {
    maxNewTokens = Math.min(maxNewTokens, 384);
  }
  if (used > totalBudget * 0.55) {
    history = trimHistoryForContext(history, Math.max(1200, Math.floor(budgets.historyChars * 0.4)), false);
    webContext = capWebContext(webContext, Math.min(webContext.length, 450));
    maxNewTokens = Math.min(maxNewTokens, 288);
    used = Object.values(breakdown()).reduce((a, b) => a + b, 0);
  }

  const staminaPercent = Math.max(0, Math.min(100, Math.round((1 - used / totalBudget) * 100)));

  return {
    history,
    webContext,
    maxNewTokens,
    usedChars: used,
    totalBudget,
    staminaPercent,
    breakdown: breakdown(),
  };
};
