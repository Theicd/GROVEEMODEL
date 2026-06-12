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

/** User asks about live camera / environment / presence / consciousness. */
export const isCameraContextQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /מה אתה רואה|מה אתה מזהה|מה מזהה|מה רואים|מה קורה|מה יש במרחב|מה יש בחדר/i.test(t) ||
    /מה השתנה|מה אני עושה|יש משהו מעניין|מה קורה סביב|מה יש ליד/i.test(t) ||
    /נוכחות|יציבות|וודאות|בטוח|קריסה|אותו מצב|המשכיות/i.test(t) ||
    /what do you see|what do you detect|what'?s happening|what is in (the|this) (room|space)/i.test(t) ||
    /presence|certainty|same (person|state)|still there/i.test(t)
  );
};

/** Consciousness / continuity / certainty questions while camera is on. */
export const isConsciousnessQuestion = (text: string): boolean => {
  const t = text.trim();
  return /נוכחות|וודאות|בטוח|קריסה|אותו|המשכיות|phantom|stable|confidence|certainty/i.test(t);
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
    (/((האם|מה).*(עומד|יושב|תנוחה))/i.test(t) && /(אדם|person|הוא|היא|\?)/i.test(t)) ||
    /מה\s*אני\s*מחזיק|what\s*am\s*i\s*holding/i.test(t) ||
    /האם\s*הוא\s*מחזיק|מה\s*(הוא|היא|אני)\s*מחזיק|what\s*(is|are)\s*(he|she|they|i)\s*holding/i.test(t) ||
    /לאן\s*(הוא|היא|אני)\s*מסתכל|where\s*(is|are)\s*(he|she|they|i)\s*looking/i.test(t) ||
    (/posture|standing|sitting/i.test(t) && /(person|אדם|\?)/i.test(t))
  );
};

/** User asks about mood/emotion of person in frame — use emotion sensor data. */
export const isPersonMoodQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /מצב\s*ה?רוח|איך\s*(הוא|היא|האדם|אני)\s*מרגיש|what\s*(is|'?s)\s*(his|her|their|your)\s*mood/i.test(t) ||
    (/emotion|mood|מרגיש|רגש/.test(t) && /(אדם|person|הוא|היא|אני|phone|טלפון)/i.test(t))
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
  isFingerCountQuestion(text) ||
  isConsciousnessQuestion(text);

/** User asks gender / age of person in frame. */
export const isPersonDemographicsQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /גבר\s*או\s*אישה|אישה\s*או\s*גבר|זכר\s*או\s*נקבה|בן\s*כמה|בת\s*כמה|מה\s*הגיל|איזה\s*גיל|how\s*old|man\s*or\s*woman|male\s*or\s*female|gender|age\s*estimate/i.test(
      t,
    ) || (/גבר|אישה|זכר|נקבה|גיל/.test(t) && /רואה|מזהה|see|detect/i.test(t))
  );
};

/** Any message that should pull live camera context into chat (text brief + optional snapshot). */
export const needsLiveCameraContext = (text: string): boolean =>
  needsCameraVisionEscalation(text) ||
  isCameraContextQuestion(text) ||
  isPersonDemographicsQuestion(text) ||
  isPersonMoodQuestion(text);

/** Remove internal vision context if the model echoed it into the answer. */
export const stripLeakedVisionReport = (text: string): string => {
  if (!text.trim()) return text;
  let out = text;
  const markers = [
    /Perception snapshot \(internal[\s\S]*?(?=\n\n[^\n-]|$)/gi,
    /═+\s*\n?PRE-CHAT VISION REPORT[\s\S]*?(?=\n\n(?![═\-])|$)/gi,
    /PRE-CHAT VISION REPORT[\s\S]*?(?=RULE:|$)/gi,
    />>> PERSON VISIBLE NOW:[\s\S]*?(?=\n\n|$)/gi,
    /\[INTERNAL VISION CONTEXT[\s\S]*?\[\/INTERNAL\]/gi,
  ];
  for (const re of markers) out = out.replace(re, "");
  out = out
    .replace(/^Person visible:[^\n]*\n/gm, "")
    .replace(/^YOLO persons:[^\n]*\n/gm, "")
    .replace(/^Faces:[^\n]*\n/gm, "")
    .replace(/^Fresh snapshot:[^\n]*\n/gm, "")
    .replace(/^Holding \(sensor\):[^\n]*\n/gm, "")
    .replace(/^Scene \(VLM\):[^\n]*\n/gm, "")
    .replace(/^HAL soul:[^\n]*\n/gm, "")
    .replace(/^-\s*Camera:\s*ACTIVE[^\n]*\n/gm, "")
    .replace(/^-\s*YOLO persons:[^\n]*\n/gm, "")
    .replace(/^-\s*Face-api faces:[^\n]*\n/gm, "")
    .replace(/^-\s*FACE DATA[^\n]*\n/gm, "")
    .replace(/^-\s*HAL soul:[^\n]*\n/gm, "")
    .replace(/^-\s*Emotion:[^\n]*\n/gm, "")
    .replace(/^-\s*Attention:[^\n]*\n/gm, "")
    .replace(/^-\s*Scene:[^\n]*\n/gm, "")
    .replace(/^RULE: Do NOT contradict[^\n]*\n/gm, "")
    .trim();
  return out;
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
  if (BORED_PLAY_RE.test(t)) return "bored_play";
  if (needsCameraVisionEscalation(t) || isCameraContextQuestion(t)) return "camera";
  if (DESIGN_RE.test(t)) return "design";
  return "general";
};

/** User wants dialogue / play / empathy / creative — NOT environment vision. */
export const isConversationFirstRequest = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    BORED_PLAY_RE.test(t) ||
    /ספר לי|סיפור|בוא נדבר|מה דעתך|what do you think|tell me about/i.test(t) ||
    /קורה לי|קרה לי|happened to me|שיתף|רוצה לשתף/i.test(t) ||
    /הפסק|חשמל|עייף|עצוב|שמח|stressed|tired/i.test(t) ||
    /רעיון|מדע בדיוני|קונספיר|fiction|story idea|creative writing/i.test(t) ||
    /תן לי|תגיד לי.*(רעיון|idea)|give me an idea/i.test(t)
  );
};

/** Explicit ask about sight, people, posture, room — needs sensors + snapshot. */
export const isExplicitVisionQuestion = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    isCameraContextQuestion(t) ||
    isPersonVisibilityQuestion(t) ||
    isCurrentPersonStateQuestion(t) ||
    isPersonDemographicsQuestion(t) ||
    isPersonMoodQuestion(t) ||
    isPersonActivityQuestion(t) ||
    isVisualDetailQuestion(t) ||
    isFingerCountQuestion(t) ||
    isConsciousnessQuestion(t)
  );
};

/** Run OCR only when user needs verbatim transcription (saves time on local machine). */
export const wantsExactTextExtraction = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /מה כתוב|מה רשום|תמלול|העתק.*טקסט|copy.*text|exact text|מדויק/i.test(t) ||
    /קרא את (כל )?הטקסט|read (all )?(the )?text/i.test(t)
  );
};

/** User attached image(s) — analyze document/photo content, not live camera. */
export const needsAttachedDocumentAnalysis = (_text: string, hasImages: boolean): boolean =>
  hasImages;

/** User wants a fillable HTML page that mirrors the photographed worksheet. */
export const wantsWorksheetReplicaHtml = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  return (
    /חלץ.*(שאלות|דף|טבלא|טקסט)|extract.*(questions|worksheet|page|text)/i.test(t) ||
    /(HTML|html|קובץ).*?(זהה|identical|same|כמו)|זהה ל(תמונה|דף|מקור|צילום)/i.test(t) ||
    /צור.*(HTML|קובץ|דף).*?(זהה|identical|למילוי|fill|print|הדפס|עבודה)/i.test(t) ||
    /דף עבודה.*HTML|worksheet.*html|HTML.*worksheet/i.test(t) ||
    /למלא.*(תשוב|דף)|fillable|fill.?in|להדפיס|לצלם שוב/i.test(t) ||
    /שחזר.*(דף|HTML)|recreate.*(page|worksheet)/i.test(t)
  );
};

/** Heuristic: user likely wants text read from an attached image. */
export const isDocumentImageRequest = (text: string): boolean => {
  const t = text.trim();
  if (!t) return true;
  return (
    /שיעורי\s*(ה)?בית|homework|worksheet|מבחן|תרגיל|exercise|מסמך|document|צילום/i.test(t) ||
    /שאל(ות|ה)|לפתור|פתור|solve|answer (the )?question/i.test(t) ||
    /תסתכל על (ה)?תמונה|על התמונה|look at (the )?(image|picture|photo)/i.test(t) ||
    /מה כתוב|קרא (את|מה)|what (does|is) (it|the image) say|read (the|this)/i.test(t) ||
    /עזרה עם|help with|תרגם|translate/i.test(t)
  );
};

/** Camera on but user talks about ideas, identity, stories — no sensor injection. */
export const isVisionUnrelatedTurn = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  if (isExplicitVisionQuestion(t)) return false;
  if (isConversationFirstRequest(t)) return true;
  if (/איך קוראים|what'?s your name|מי אתה|who are you/i.test(t)) return true;
  if (/רעיון|סיפור|מדע בדיונ|קונספיר|ברמודה|פירמיד|ירח|fiction|plot|character/i.test(t)) return true;
  if (/תן לי|תגיד לי|help me write|brainstorm/i.test(t) && !/רואה|see|מצלמה|camera/i.test(t)) return true;
  if (isCodeGenerationRequest(t)) return true;
  if (isSimpleGreeting(t) && !/רואה|see|מזהה|detect/i.test(t)) return true;
  return false;
};

/** Attach snapshot + sensor report to this turn. */
export const needsVisionSensorContext = (text: string): boolean => {
  if (isVisionUnrelatedTurn(text)) return false;
  return isExplicitVisionQuestion(text) || isSceneInterpretationQuestion(text);
};

export const formatCameraTopicLabel = (topic: string): string => {
  const key = topic.trim();
  const map: Record<string, string> = {
    greeting: "ברכה",
    camera: "ראייה",
    general: "שיחה",
    design: "עיצוב",
    bored_play: "משחק / שעמום",
    scene_general: "סצנה",
    visibility: "נוכחות",
    person: "אדם בפריים",
    mood: "מצב רוח",
  };
  if (key.startsWith("pack:")) return "רגע / מצב";
  if (key.startsWith("situation:")) return "מצב";
  return map[key] ?? key.replace(/^(pack|topic):/, "");
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
  const cleaned =
    cleanDisplayText(base) ||
    (thought ? cleanDisplayText(thought) : cleanDisplayText(raw));
  return stripLeakedVisionReport(cleaned);
};
