import { buildTriviaFormatInstruction } from "./trivia/triviaPrompt";
import { formatSessionMemoryForPrompt } from "./chatSessionMemory";
import { capWebContext } from "./chatResourceBudget";
import type { ChatTurnPreludeContinue, PendingWebSearchMeta } from "./chatTurnPrelude";
import {
  DEFAULT_LOCAL_TEXT_SETTINGS,
  localTextBaseSystemForUi,
  type LocalTextModelSettings,
} from "./modelRack/localTextModelSettings";
import { SMOLLM_GROVEE_IDENTITY } from "./modelRack/localTextTranslate";
import type { ChatUiLanguage } from "./ui/useUiLanguage";
import type { StartupContext } from "./startupContext";
import { personalityBlockForSmolLM } from "./personalityProfile";
import { imageDescribeSystemAppend } from "./imageGen/imageDescribeHint";

/** Small SmolLM models collapse on long Gemma-style system blocks — keep this tiny. */
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
  /** User "remember …" facts extracted from the session. */
  sessionMemoryFacts?: string[];
  triviaMode?: boolean;
  triviaQuestionCount?: number;
  documentContext?: string;
  userName?: string;
};

/** Trim append blocks first; never truncate GROVEE identity core. */
export function trimLocalTextSystemPrompt(base: string, appendBlocks: string[]): string {
  const core = base.trim();
  const parts = appendBlocks.map((b) => b.trim()).filter(Boolean);
  let combined = core;
  for (const block of parts) {
    const next = `${combined}\n\n${block}`;
    if (next.length <= LOCAL_TEXT_MAX_SYSTEM_CHARS) {
      combined = next;
      continue;
    }
    const room = LOCAL_TEXT_MAX_SYSTEM_CHARS - combined.length - 2;
    if (room > 40) {
      combined = `${combined}\n\n${block.slice(0, room - 1)}…`;
    }
    break;
  }
  if (combined.length <= LOCAL_TEXT_MAX_SYSTEM_CHARS) return combined;
  const identityLen = Math.min(core.length, SMOLLM_GROVEE_IDENTITY.length + 80);
  const head = core.slice(0, identityLen).trim();
  const tailRoom = LOCAL_TEXT_MAX_SYSTEM_CHARS - head.length - 1;
  return tailRoom > 0 ? `${head.slice(0, tailRoom)}…` : head.slice(0, LOCAL_TEXT_MAX_SYSTEM_CHARS - 1) + "…";
}

export function buildLocalTextSystemPrompt(input: LocalTextContextInput): string {
  const settings = input.settings ?? DEFAULT_LOCAL_TEXT_SETTINGS;
  const { uiLang, prelude, pendingWebSearch, webContext } = input;
  const baseSystem = localTextBaseSystemForUi(uiLang, settings);
  const appendBlocks: string[] = [];

  if (prelude.greeting) {
    return uiLang === "he"
      ? "You are Groovie. Reply with one short friendly Hebrew greeting sentence only."
      : "You are Groovie. Reply with one short friendly greeting sentence only.";
  }

  const searchHadLiveData =
    pendingWebSearch?.sources.some((s) => s.ok && s.text.trim()) ?? false;

  const cappedWeb = capWebContext(webContext, Math.min(settings.webBriefChars, 420));
  if (cappedWeb.trim() && searchHadLiveData) {
    appendBlocks.push(`Use only these live facts:\n${cappedWeb}`);
  } else if (prelude.shouldRunWebSearch && !searchHadLiveData && !prelude.conversationalTurn) {
    appendBlocks.push(
      "Live search returned no usable data. Say briefly that live fetch failed. Do not invent facts.",
    );
  } else if (prelude.conversationalTurn && !searchHadLiveData) {
    appendBlocks.push(
      uiLang === "he"
        ? "ענה בשיחה חופשית וקצרה — זו שאלת דעה/יצירתיות, לא בקשת מידע חי."
        : "Reply conversationally and briefly — opinion or creative chat, not a live-data lookup.",
    );
  }

  if (prelude.gameNoResults) {
    appendBlocks.push("No games matched. Tell the user briefly and point to category browse.");
  } else if (prelude.gameGroundingBlock.trim()) {
    appendBlocks.push(
      `Games are shown inline in chat — user taps ▶ on cards:\n${prelude.gameGroundingBlock.slice(0, 200)}`,
    );
  }

  if (prelude.globePlaceLabel) {
    appendBlocks.push(`Map focus: ${prelude.globePlaceLabel}`);
  }

  if (prelude.localTimeOnly) {
    appendBlocks.push("Answer using local time context only.");
  }

  const memoryBlock = formatSessionMemoryForPrompt(input.sessionMemoryFacts ?? []);
  if (memoryBlock) appendBlocks.push(memoryBlock);

  if (input.userName?.trim()) {
    appendBlocks.push(`User name: ${input.userName.trim()}. Use naturally when appropriate.`);
  }

  if (input.documentContext?.trim()) {
    appendBlocks.push(input.documentContext.trim().slice(0, 520));
  }

  if (prelude.imageDescribeMode) {
    appendBlocks.push(imageDescribeSystemAppend(uiLang));
  }

  appendBlocks.push(personalityBlockForSmolLM(uiLang));

  if (input.triviaMode) {
    appendBlocks.push(
      buildTriviaFormatInstruction(uiLang, input.triviaQuestionCount ?? 5),
    );
  }

  return trimLocalTextSystemPrompt(baseSystem, appendBlocks);
}

export function localTextMaxNewTokens(
  prelude: ChatTurnPreludeContinue,
  settings: LocalTextModelSettings = DEFAULT_LOCAL_TEXT_SETTINGS,
): number {
  if (prelude.triviaMode) return Math.min(896, Math.max(settings.maxNewTokensSearch, 640));
  if (prelude.shouldRunWebSearch || prelude.localTimeOnly) {
    return Math.min(settings.maxNewTokensSearch, 320);
  }
  if (prelude.greeting) return settings.maxNewTokensGreeting;
  return settings.maxNewTokens;
}
