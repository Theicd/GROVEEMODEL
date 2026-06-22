import {
  buildWebSearchGroundingAppend,
  GAME_SEARCH_GROUNDING_APPEND,
  GAME_SEARCH_NO_RESULTS_APPEND,
  WEB_SEARCH_NO_RESULTS_APPEND,
} from "./characterPrompts";
import { GLOBE_PRESENTATION_APPEND } from "./realityGlobe/globePresentation";
import { capWebContext } from "./chatResourceBudget";
import { buildStartupPromptBlock, type StartupContext } from "./startupContext";
import type { ChatTurnPreludeContinue, PendingWebSearchMeta } from "./chatTurnPrelude";
import {
  DEFAULT_LOCAL_TEXT_SETTINGS,
  localTextBaseSystemForUi,
  type LocalTextModelSettings,
} from "./modelRack/localTextModelSettings";
import type { ChatUiLanguage } from "./ui/useUiLanguage";

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

export function buildLocalTextSystemPrompt(input: LocalTextContextInput): string {
  const settings = input.settings ?? DEFAULT_LOCAL_TEXT_SETTINGS;
  const { uiLang, prelude, pendingWebSearch, startupContext, webContext } = input;
  let systemPrompt = localTextBaseSystemForUi(uiLang, settings);

  if (startupContext) {
    systemPrompt = `${systemPrompt}\n\n${buildStartupPromptBlock(startupContext)}`;
  }

  const searchHadLiveData =
    pendingWebSearch?.sources.some((s) => s.ok && s.text.trim()) ?? false;
  if (searchHadLiveData) {
    systemPrompt = `${systemPrompt}\n\n${buildWebSearchGroundingAppend({
      answerShape: pendingWebSearch?.answerShape,
      crossSource: pendingWebSearch?.crossSource,
    })}`;
  } else if (prelude.shouldRunWebSearch) {
    systemPrompt = `${systemPrompt}\n\n${WEB_SEARCH_NO_RESULTS_APPEND}`;
  }

  if (prelude.gameNoResults) {
    systemPrompt = `${systemPrompt}\n\n${GAME_SEARCH_NO_RESULTS_APPEND}`;
  } else if (prelude.gameGroundingBlock.trim()) {
    systemPrompt = `${systemPrompt}\n\n${GAME_SEARCH_GROUNDING_APPEND}\nGames found:\n${prelude.gameGroundingBlock}`;
  }

  if (prelude.globePlaceLabel) {
    systemPrompt = `${systemPrompt}\n\n${GLOBE_PRESENTATION_APPEND}\nPlace shown on map: ${prelude.globePlaceLabel}`;
  }

  const cappedWeb = capWebContext(webContext, settings.webBriefChars);
  if (cappedWeb.trim()) {
    systemPrompt = `${systemPrompt}\n\n[WEB CONTEXT — ground truth for this turn]\n${cappedWeb}`;
  }

  if (prelude.greeting) {
    systemPrompt = `${systemPrompt}\n\nIf the user sends only a greeting, reply with one short warm sentence.`;
  }

  return systemPrompt;
}

export function localTextMaxNewTokens(
  prelude: ChatTurnPreludeContinue,
  settings: LocalTextModelSettings = DEFAULT_LOCAL_TEXT_SETTINGS,
): number {
  if (prelude.shouldRunWebSearch || prelude.localTimeOnly) {
    return settings.maxNewTokensSearch;
  }
  if (prelude.greeting) return settings.maxNewTokensGreeting;
  return settings.maxNewTokens;
}
