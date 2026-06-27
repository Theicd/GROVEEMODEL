import { capWebContext } from "./chatResourceBudget";
import type { ChatTurnPreludeContinue, PendingWebSearchMeta } from "./chatTurnPrelude";
import {
  DEFAULT_LOCAL_TEXT_SETTINGS,
  localTextBaseSystemForUi,
  type LocalTextModelSettings,
} from "./modelRack/localTextModelSettings";
import type { ChatUiLanguage } from "./ui/useUiLanguage";
import type { StartupContext } from "./startupContext";

/** SmolLM 360M collapses on long Gemma-style system blocks — keep this tiny. */
export const LOCAL_TEXT_MAX_SYSTEM_CHARS = 900;

/** @deprecated use settings.webBriefChars */
export const LOCAL_TEXT_WEB_BRIEF_CHARS = DEFAULT_LOCAL_TEXT_SETTINGS.webBriefChars;

/** @deprecated use settings.maxNewTokensSearch */
export const LOCAL_TEXT_SEARCH_MAX_TOKENS = DEFAULT_LOCAL_TEXT_SETTINGS.maxNewTokensSearch;

/** @deprecated use settings.maxNewTokens */
export const LOCAL_TEXT_DEFAULT_MAX_TOKENS = DEFAULT_LOCAL_TEXT_SETTINGS.maxNewTokens;

export type LocalTextContextInput = {
  uiLang: ChatUiLanguage;
  prelude: ChatTurnPreludeContinue;
  pendingWebSearch: PendingWebSearchMeta | null;
  startupContext: StartupContext | null;
  webContext: string;
  settings?: LocalTextModelSettings;
};

function trimSystemPrompt(text: string): string {
  const clean = text.trim();
  if (clean.length <= LOCAL_TEXT_MAX_SYSTEM_CHARS) return clean;
  return `${clean.slice(0, LOCAL_TEXT_MAX_SYSTEM_CHARS - 1)}…`;
}

export function buildLocalTextSystemPrompt(input: LocalTextContextInput): string {
  const settings = input.settings ?? DEFAULT_LOCAL_TEXT_SETTINGS;
  const { uiLang, prelude, pendingWebSearch, webContext } = input;
  let systemPrompt = localTextBaseSystemForUi(uiLang, settings);

  if (prelude.greeting) {
    return trimSystemPrompt(
      `${systemPrompt}\nThe user sent a short greeting. Reply with one warm sentence only.`,
    );
  }

  const searchHadLiveData =
    pendingWebSearch?.sources.some((s) => s.ok && s.text.trim()) ?? false;

  const cappedWeb = capWebContext(webContext, Math.min(settings.webBriefChars, 420));
  if (cappedWeb.trim() && searchHadLiveData) {
    systemPrompt = `${systemPrompt}\n\nUse only these live facts:\n${cappedWeb}`;
  } else if (prelude.shouldRunWebSearch && !searchHadLiveData) {
    systemPrompt = `${systemPrompt}\n\nLive search returned no usable data. Say briefly that live fetch failed. Do not invent facts.`;
  }

  if (prelude.gameNoResults) {
    systemPrompt = `${systemPrompt}\n\nNo games matched. Tell the user briefly and point to category browse.`;
  } else if (prelude.gameGroundingBlock.trim()) {
    systemPrompt = `${systemPrompt}\n\nGames are listed in the side panel only:\n${prelude.gameGroundingBlock.slice(0, 240)}`;
  }

  if (prelude.globePlaceLabel) {
    systemPrompt = `${systemPrompt}\n\nMap focus: ${prelude.globePlaceLabel}`;
  }

  if (prelude.localTimeOnly) {
    systemPrompt = `${systemPrompt}\n\nAnswer using local time context only.`;
  }

  return trimSystemPrompt(systemPrompt);
}

export function localTextMaxNewTokens(
  prelude: ChatTurnPreludeContinue,
  settings: LocalTextModelSettings = DEFAULT_LOCAL_TEXT_SETTINGS,
): number {
  if (prelude.shouldRunWebSearch || prelude.localTimeOnly) {
    return Math.min(settings.maxNewTokensSearch, 320);
  }
  if (prelude.greeting) return settings.maxNewTokensGreeting;
  return settings.maxNewTokens;
}
