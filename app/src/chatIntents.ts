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

/** User asks about live camera / environment context. */
export const isCameraContextQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /מה אתה רואה|מה אתה רואה\?|מה השתנה|מה אני עושה|יש משהו מעניין|מה קורה סביב|מה יש ליד/i.test(
      t,
    ) ||
    /what do you see|what changed|what am i doing|anything interesting|what'?s happening/i.test(t)
  );
};

/** User asks about a specific visual detail — requires fresh snapshot, not memory alone. */
export const isVisualDetailQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /מה (ה)?שעה|מה השעה בשעון|what time|what's the time|what is the time/i.test(t) ||
    /מה כתוב|מה רשום|what (does it|is) say|what is written|what's written|what does the .+ say/i.test(t) ||
    /איזה צבע|what colou?r/i.test(t) ||
    /כמה אנשים|how many people/i.test(t) ||
    /איזה דגם|what model|what brand|what kind of/i.test(t) ||
    /מה רואים על המסך|what'?s on the (screen|monitor)|על המסך|on the screen/i.test(t) ||
    /(השעון|החולצה|המסך|הגיטרה|the clock|the shirt|the guitar|the screen)/i.test(t) &&
      /(מה|איזה|how|what|כמה|\?)/i.test(t)
  );
};

/** User asks if the character sees them — needs snapshot + honest people answer. */
export const isPersonVisibilityQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /אתה רואה אותי|את רואה אותי|רואה אותי|רואה אותי\?|אתה רואה אותי\?/i.test(t) ||
    /do you see me|can you see me|am i visible|are you seeing me/i.test(t)
  );
};

/** User asks what someone is doing — interpret activity, not caption. */
export const isPersonActivityQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /מה (ה)?אדם עושה|מה הוא עושה|מה היא עושה|מה האדם עושה|מה קורה שם|what is (he|she|the person|they) doing|what are they doing/i.test(
      t,
    ) ||
    (/עושה עכשיו|doing now|right now/i.test(t) && /(אדם|person|הוא|היא|they)/i.test(t))
  );
};

/** Posture / holding / gaze — requires fresh person focus, not stale memory. */
export const isCurrentPersonStateQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /עומד\s*או\s*יושב|יושב\s*או\s*עומד|standing\s*or\s*sitting/i.test(t) ||
    /האדם\s*(עומד|יושב)|האם\s*(הוא|היא|האדם)\s*(עומד|יושב)/i.test(t) ||
    /(האם|מה).*(עומד|יושב|תנוחה)/i.test(t) && /(אדם|person|הוא|היא|\?)/i.test(t) ||
    /האם\s*הוא\s*מחזיק|מה\s*(הוא|היא)\s*מחזיק|what\s*(is|are)\s*(he|she|they)\s*holding/i.test(t) ||
    /לאן\s*(הוא|היא)\s*מסתכל|where\s*(is|are)\s*(he|she|they)\s*looking/i.test(t) ||
    (/posture|standing|sitting/i.test(t) && /(person|אדם|\?)/i.test(t))
  );
};

/** User asks how many fingers are visible — needs fresh hand sensor data. */
export const isFingerCountQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /כמה אצבעות|כמה אצבע|כמה אצבעות אתה רואה|כמה אצבעות את רואה/i.test(t) ||
    /how many finger|how many fingers/i.test(t)
  );
};

export const needsPersonFocusRefresh = (text: string): boolean =>
  isCurrentPersonStateQuestion(text) ||
  isPersonActivityQuestion(text) ||
  isPersonVisibilityQuestion(text) ||
  isFingerCountQuestion(text);

/** Scene questions needing interpretation, not inventory (excludes factual visual-detail). */
export const isSceneInterpretationQuestion = (text: string): boolean => {
  if (isVisualDetailQuestion(text)) return false;
  return (
    isCameraContextQuestion(text) ||
    isPersonActivityQuestion(text) ||
    /איך (ה)?חדר|מה (ה)?אווירה|what'?s the mood|how does (the|it) (room|space) feel/i.test(text.trim())
  );
};

/** Camera mode: attach live snapshot + vision model for this message. */
export const needsCameraVisionEscalation = (text: string): boolean =>
  isSceneInterpretationQuestion(text) ||
  isVisualDetailQuestion(text) ||
  isPersonVisibilityQuestion(text) ||
  isCurrentPersonStateQuestion(text) ||
  isFingerCountQuestion(text);

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

/** Conversational topic buckets for shift detection (chat layer, not vision). */
export type ChatTopic =
  | "greeting"
  | "design"
  | "bored_play"
  | "camera"
  | "general";

const DESIGN_RE =
  /עיצוב|כסא|כיסא|חומר|מינימל|אווירה|חדר|ריהוט|סגנון|design|chair|material|minimal|room decor|interior/i;
const BORED_PLAY_RE =
  /משעמם|משחק|נשחק|\bbored\b|play a game|let'?s play|מה אתה מציע|מה להציע|kill time|entertain/i;

export const classifyChatTopic = (text: string): ChatTopic => {
  const t = text.trim();
  if (!t) return "general";
  if (isSimpleGreeting(t)) return "greeting";
  if (needsCameraVisionEscalation(t) || isCameraContextQuestion(t)) return "camera";
  if (BORED_PLAY_RE.test(t)) return "bored_play";
  if (DESIGN_RE.test(t)) return "design";
  return "general";
};

/** True when the user clearly moved to a different conversational lane. */
export const isTopicShift = (prev: ChatTopic | null, next: ChatTopic): boolean => {
  if (!prev || prev === next) return false;
  if (prev === "greeting" && next === "general") return false;
  if (prev === "general" || next === "general") return prev !== next && next !== "greeting";
  return prev !== next;
};

export const topicShiftHint = (from: ChatTopic, to: ChatTopic): string => {
  const fromLabel =
    from === "design" ? "room/design" : from === "bored_play" ? "bored/play" : from;
  const toLabel =
    to === "design" ? "room/design" : to === "bored_play" ? "bored/play" : to;
  return `TOPIC SHIFT: The user moved from "${fromLabel}" to "${toLabel}". Respond ONLY to their new intent. Do NOT continue the previous thread (e.g. do not mention chairs, materials, or room design if they asked to play or said they are bored).`;
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
