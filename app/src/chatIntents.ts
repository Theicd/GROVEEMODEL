/** Pure helpers for routing user text (used by App and unit tests). */

export type ChatTurnImageRef = { bytes: ArrayBuffer; mime: string };

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  images?: ChatTurnImageRef[];
};

export const isSimpleGreeting = (text: string): boolean => {
  const normalized = text.trim().toLowerCase();
  return /^(hi|hey|hello|shalom|שלום|היי|הי)$/.test(normalized);
};

export const isRtlText = (text: string): boolean => /[\u0590-\u05FF]/.test(text);

/** User asks to continue a previous (cut-off) answer. */
export const isContinueRequest = (text: string): boolean => {
  const t = text.trim().toLowerCase();
  if (/^(continue|go on|keep going|resume)\b/.test(t)) return true;
  if (/^(המשך|תמשיך|המשיכי|המשך לכתוב|תמשיך לכתוב|המשך מהמקום|המשך את הקוד)/.test(t)) return true;
  if (/continue\s+(writing|the\s+code|from)/.test(t)) return true;
  if (/המשך.*(קוד|html|כתיב)/.test(t)) return true;
  return false;
};

export const isCodeGenerationRequest = (text: string): boolean => {
  const t = text.toLowerCase();
  return (
    /```/.test(text) ||
    /\b(html|css|javascript|typescript|python|webgl|react|vue|node|sql)\b/.test(t) ||
    /(קוד|html|webgl|סקריפט|דף\s*html|קובץ\s*html)/i.test(text)
  );
};

/** Odd number of ``` fences → output stopped inside a code block. */
export const hasUnclosedCodeFence = (content: string): boolean => {
  const fences = content.match(/```/g);
  if (!fences) return false;
  return fences.length % 2 !== 0;
};

export const lastAssistantTurn = (turns: ChatTurn[]): ChatTurn | undefined => {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === "assistant") return turns[i];
  }
  return undefined;
};

export const shouldContinueCode = (userText: string, turns: ChatTurn[]): boolean => {
  if (!isContinueRequest(userText)) return false;
  const last = lastAssistantTurn(turns);
  if (!last) return false;
  return hasUnclosedCodeFence(last.content) || isCodeGenerationRequest(last.content);
};

/** Keep recent turns within a char budget; optionally always keep the last assistant reply. */
export const trimHistoryForContext = (
  turns: ChatTurn[],
  maxChars = 32_000,
  pinLastAssistant = false,
): ChatTurn[] => {
  if (turns.length === 0) return [];

  let lastAssistantIdx = -1;
  if (pinLastAssistant) {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }
  }

  const picked: ChatTurn[] = [];
  let budget = maxChars;

  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i];
    const cost = turn.content.length + 64;

    if (i === lastAssistantIdx) {
      picked.unshift(turn);
      budget -= cost;
      continue;
    }

    if (cost <= budget) {
      picked.unshift(turn);
      budget -= cost;
    } else {
      break;
    }
  }

  return picked;
};

export const CONTINUE_CODE_SYSTEM_HINT =
  "CRITICAL: Your previous assistant reply was CUT OFF mid-code (token limit). The user wants you to CONTINUE from exactly where you stopped. Do NOT restart. Do NOT ask the user to show the file or repeat the question. Output ONLY the continuation text — it may start mid-line or mid-tag. If you were inside a ``` code fence, continue inside it without opening a duplicate fence unless you already closed the previous one.";

export const CODE_TOKEN_FLOOR = 1536;
export const CODE_TOKEN_CAP = 2048;

/** Split Gemma 4 native thinking output into thought channel vs final answer. */
export const parseGemmaThinkingOutput = (
  raw: string,
): { thought: string; answer: string; hasThinking: boolean } => {
  const text = raw.replace(/\r/g, "");
  const thoughtMarker = "<|channel>thought";
  let idx = text.indexOf(thoughtMarker);

  if (idx === -1 && /^thought\b/im.test(text.trimStart())) {
    idx = text.search(/^thought\b/im);
  }

  if (idx === -1) {
    return { thought: "", answer: text, hasThinking: false };
  }

  let afterMarker = text.slice(idx);
  if (afterMarker.startsWith(thoughtMarker)) {
    afterMarker = afterMarker.slice(thoughtMarker.length).replace(/^\s*\n?/, "");
  } else {
    afterMarker = afterMarker.replace(/^thought\b\s*\n?/i, "");
  }

  const endMatch = afterMarker.match(/\n\s*\n/);
  if (!endMatch || endMatch.index === undefined) {
    return { thought: afterMarker.trim(), answer: "", hasThinking: true };
  }

  const thought = afterMarker.slice(0, endMatch.index).trim();
  const answer = afterMarker.slice(endMatch.index + endMatch[0].length).trim();
  return { thought, answer, hasThinking: true };
};

export type AssistantStreamParts = {
  thought: string;
  answer: string;
  /** Model is still streaming the thought section (no answer/code yet). */
  thinkingInProgress: boolean;
};

/** Index where answer code/HTML begins (line-start fence or document), not inline mentions in prose. */
export const findAnswerContentStart = (text: string): number => {
  const patterns = [
    /(?:^|\n)\s*```html\b/im,
    /(?:^|\n)\s*```[\w-]*\s*\n/im,
    /(?:^|\n)\s*<!DOCTYPE\s+html/im,
    /(?:^|\n)\s*<html\b/im,
  ];
  let best = -1;
  for (const re of patterns) {
    const m = text.match(re);
    if (m?.index !== undefined) {
      const pos = m.index + (m[0].startsWith("\n") ? 1 : 0);
      if (best === -1 || pos < best) best = pos;
    }
  }
  return best;
};

/**
 * Split streaming assistant output into thought vs answer.
 * When Think is on, artifact detection and code preview must use `answer` only.
 */
export const splitAssistantStream = (raw: string, thinkingEnabled: boolean): AssistantStreamParts => {
  const text = raw.replace(/\r/g, "");

  const native = parseGemmaThinkingOutput(text);
  if (native.hasThinking) {
    if (native.answer.trim()) {
      return { thought: native.thought, answer: native.answer, thinkingInProgress: false };
    }
    return { thought: native.thought || text.trim(), answer: "", thinkingInProgress: true };
  }

  if (!thinkingEnabled) {
    return { thought: "", answer: text, thinkingInProgress: false };
  }

  const codeStart = findAnswerContentStart(text);
  if (codeStart >= 0) {
    return {
      thought: text.slice(0, codeStart).trim(),
      answer: text.slice(codeStart).trim(),
      thinkingInProgress: false,
    };
  }

  const looksLikeThought =
    /^<\|think\|>/m.test(text) ||
    /^thought\b/im.test(text.trimStart()) ||
    /thinking process:/i.test(text);

  if (looksLikeThought) {
    return { thought: text.trim(), answer: "", thinkingInProgress: true };
  }

  return { thought: "", answer: text, thinkingInProgress: false };
};

/** Portion of the stream to scan for HTML/code artifacts (excludes thought preamble). */
export const getArtifactScanContent = (raw: string, thinkingEnabled: boolean): string => {
  const { answer, thinkingInProgress } = splitAssistantStream(raw, thinkingEnabled);
  if (thinkingInProgress) return "";
  return answer;
};

/** Strip Gemma control tokens without splitting thought/answer (for display fields). */
export const cleanDisplayText = (raw: string): string =>
  raw
    .replace(/<\|channel>[^<\n]*/g, "")
    .replace(/<\|think\|>/g, "")
    .replace(/<\|turn>[^<\n]*/g, "")
    .replace(/<\|[^|>]+>/g, "")
    .replace(/\r/g, "")
    .split("\n")
    .filter((line) => !/^\s*(User|Assistant|System|model)\s*:/i.test(line))
    .join("\n")
    .replace(/^["']+|["']+$/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

/** Strip Gemma control tokens and return user-visible answer text. */
export const stripGemmaControlTokens = (raw: string): string => {
  const { answer, thought, hasThinking } = parseGemmaThinkingOutput(raw);
  const base = hasThinking && answer ? answer : raw;
  return (
    cleanDisplayText(base) ||
    (thought ? cleanDisplayText(thought) : cleanDisplayText(raw))
  );
};
