/** User instruction line before pasted payload (prompt + optional short follow-up). */
export const extractUserIntentPrefix = (text: string): string => {
  const t = text.trim();
  if (!t) return "";
  const firstParagraph = t.split(/\n\s*\n/)[0]?.trim() ?? t;
  const lines = firstParagraph.split(/\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length <= 1) {
    return firstParagraph.length <= 180 ? firstParagraph : firstParagraph.slice(0, 180);
  }
  const firstLine = lines[0] ?? firstParagraph;
  return firstLine.length <= 180 ? firstLine : firstLine.slice(0, 180);
};

/** Word boundary that works after Hebrew (JS \\b is unreliable for Hebrew). */
const AFTER_VERB = String.raw`(?:\s|$|[?!.:,"«»])`;

const TEXT_COMPOSITION_RE = new RegExp(
  `^(?:נסח|עדכן|ערוך|כתוב|תכתוב|תרגם|סכם|שפר|הפוך|המר|המיר|ארגן|מיין|תנסח|תנסחי|תערוך|תכין|formulate|rephrase|rewrite|summarize|edit|compose|polish|improve|convert|transform)${AFTER_VERB}`,
  "i",
);

const TEXT_ANALYSIS_PREFIX_RE = new RegExp(
  `^(?:קרא|נתח|פרש|הערך|זהה|בדוק|הסבר|תסביר|תגיד|ספר|analyze|analyse|read|review|evaluate|interpret|explain)${AFTER_VERB}`,
  "i",
);

const TEXT_ANALYSIS_TASK_RE =
  /(?:מה\s+(?:ה)?טון|איז(?:ה|ו)\s+טון|what\s+(?:is\s+)?the\s+tone|sentiment|רגש(?:ות)?|כוונ(?:ה|ת)|משמעות|מסקנ(?:ה|ות)|איך\s+הגע(?:ת|תי)|how\s+did\s+you\s+(?:conclude|determine|reach)|למה\s+(?:ה)?(?:כ(?:תב|ותב)|חוש(?:ב|בת)))/i;

const TEXT_TRANSFORM_TASK_RE =
  /(?:לרשימ(?:ה|ת)|רשימ(?:ה|ת)\s+(?:של\s+)?(?:נקוד|bullet)|בעד\s+ואן|בעד\s+ונגד|for\s+and\s+against|pros?\s+and\s+cons|נקודות\s+(?:קצר|עיקר)|bullet\s*points?|מאמר\s+(?:ה)?דעה)/i;

const PASTED_PAYLOAD_RE =
  /[""«][^""«\n]{40,}|[""«][^""«\n]{40,}["»]|:\s*[""«][^""«\n]{20,}/;

/** Message includes a substantial quoted / pasted block beyond the instruction line. */
export const hasPastedTextPayload = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  const prefix = extractUserIntentPrefix(t);
  if (t.length > prefix.length + 100) return true;
  if (PASTED_PAYLOAD_RE.test(t)) return true;
  return t.split(/\n\s*\n/).length > 1 && t.length > 120;
};

/** Writing/editing intent — must not trigger live search, games panel, or canned data replies. */
export const isTextCompositionRequest = (text: string): boolean => {
  const prefix = extractUserIntentPrefix(text);
  if (!prefix) return false;
  if (TEXT_COMPOSITION_RE.test(prefix)) return true;
  return /^(?:נסח|עדכן|ערוך|כתוב|תרגם|סכם|שפר)\s+(?:את|לי|מחדש|במילים|טקסט|הודעה|מייל|prompt|הוראה)/i.test(
    prefix,
  );
};

/** Restructure pasted content (lists, pros/cons, bullet points). */
export const isTextTransformRequest = (text: string): boolean => {
  const t = text.trim();
  const prefix = extractUserIntentPrefix(t);
  if (!prefix || !hasPastedTextPayload(t)) return false;
  if (TEXT_COMPOSITION_RE.test(prefix)) return true;
  if (TEXT_TRANSFORM_TASK_RE.test(prefix)) return true;
  return /^(?:הפוך|המר|המיר|ארגן|מיין|convert|transform|turn\s+into)/i.test(prefix);
};

/** Read/analyze pasted letter, email, or paragraph — chat-only, no web search. */
export const isPastedTextAnalysisRequest = (text: string): boolean => {
  const t = text.trim();
  if (!t) return false;
  const prefix = extractUserIntentPrefix(t);
  if (!prefix) return false;

  const analysisInstruction =
    TEXT_ANALYSIS_PREFIX_RE.test(prefix) ||
    TEXT_ANALYSIS_TASK_RE.test(prefix) ||
    /^(?:קרא|נתח).*(?:ה?(?:מכתב|טקסט|הודעה|מייל|פסק(?:ה|ות))|the\s+(?:letter|email|text|message))/i.test(
      prefix,
    );

  if (!analysisInstruction) return false;

  if (!hasPastedTextPayload(t)) {
    return (
      TEXT_ANALYSIS_TASK_RE.test(prefix) &&
      /(?:ה?(?:מכתב|טקסט|הודעה|מייל)|the\s+(?:letter|email|text))/i.test(prefix)
    );
  }

  return true;
};

/**
 * Chat-only turn: compose/edit, transform, or analyze pasted text —
 * never auto web search / canned live data.
 */
export const isInlineTextTaskRequest = (text: string): boolean =>
  isTextCompositionRequest(text) ||
  isPastedTextAnalysisRequest(text) ||
  isTextTransformRequest(text);

/** True when user explicitly wants fresh / live external data (not inline text work). */
export const hasExplicitLiveDataIntent = (prefix: string): boolean =>
  /(?:חפש|חיפוש|תחפש|מצא|מצאי|search\s+for|look\s+up|עכשיו|כרגע|היום|מחיר|price|עדכנ(?:י|ים)|live|real.?time|מה\s+קור(?:ה|ה)|מה\s+המצב|what'?s\s+happening)/i.test(
    prefix,
  );

/**
 * For intent regexes: scan instruction line only when message is inline text work
 * or pasted payload without explicit live-search intent.
 */
export const getIntentScanText = (
  text: string,
  opts?: { userRequestsSearch?: (t: string) => boolean },
): string => {
  const t = text.trim();
  if (!t) return t;
  if (isInlineTextTaskRequest(t)) return extractUserIntentPrefix(t);
  const prefix = extractUserIntentPrefix(t);
  const wantsSearch = opts?.userRequestsSearch?.(t) ?? false;
  if (hasPastedTextPayload(t) && !wantsSearch && !hasExplicitLiveDataIntent(prefix)) {
    return prefix;
  }
  return t;
};
