/** Regex intent for in-chat cloud image generation (no model switch). */

export type ChatTurnLike = { role: string; content: string };

export type GeneratedImagePayload = { url: string; prompt: string };

const DESCRIBE_RE =
  /^(?:תאר|תארי|תארו|describe|paint|imagine|visualize)\s+(?:לי\s+|לי)?(?:סצנ(?:ה|ת)|תמונ(?:ה|ת)|את\s+)?(.+)/is;

const GENERATE_RE =
  /^(?:צור|צרי|צרו|תייצר|תייצרי|generate|create|draw|render|make)\s+(?:לי\s+)?(?:תמונ(?:ה|ות)|image|picture|photo)/i;

const GENERATE_WITH_SUBJECT_RE =
  /^(?:צור|צרי|צרו|תייצר|תייצרי|generate|create|draw|render|make)\s+(?:לי\s+)?(?:תמונ(?:ה|ות)|image|picture|photo)\s+(?:של\s+|of\s+|about\s+)?(.+)/is;

const FROM_PREVIOUS_RE =
  /(?:מזה|מהתיאור|לפי\s+(?:ה)?תיאור|from\s+(?:that|the\s+description)|based\s+on\s+(?:that|what))/i;

function cleanImageSubject(raw: string): string {
  return raw
    .trim()
    .replace(/^(?:של|את|of|about)\s+/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function isImageDescribeRequest(text: string): boolean {
  const t = text.trim();
  if (!t || t.length > 280) return false;
  if (GENERATE_RE.test(t) && !DESCRIBE_RE.test(t)) return false;
  return DESCRIBE_RE.test(t);
}

export function extractImageDescribeSubject(text: string): string | null {
  const m = text.trim().match(DESCRIBE_RE);
  const subject = m?.[1] ? cleanImageSubject(m[1]) : "";
  return subject.length >= 2 ? subject : null;
}

export function extractImageGenerateSubject(text: string): string | null {
  const m = text.trim().match(GENERATE_WITH_SUBJECT_RE);
  const subject = m?.[1] ? cleanImageSubject(m[1]) : "";
  return subject.length >= 2 ? subject : null;
}

export function isImageGenerateRequest(text: string): boolean {
  const t = text.trim();
  if (!t || t.length > 160) return false;
  return GENERATE_RE.test(t) || FROM_PREVIOUS_RE.test(t);
}

export function isImageFromPreviousRequest(text: string): boolean {
  return FROM_PREVIOUS_RE.test(text.trim());
}

/** Resolve English-friendly prompt for Pollinations from pending ref or prior assistant turn. */
export function resolveImagePromptFromHistory(
  text: string,
  pendingPrompt: string | null,
  priorTurns: ChatTurnLike[],
): string | null {
  const pending = pendingPrompt?.trim() || null;
  const generateSubject = extractImageGenerateSubject(text);
  if (generateSubject) return generateSubject;

  const describeSubject = extractImageDescribeSubject(text);
  if (describeSubject && isImageDescribeRequest(text)) return describeSubject;

  if (isImageFromPreviousRequest(text)) {
    if (pending) return pending;
    for (let i = priorTurns.length - 1; i >= 0; i--) {
      const t = priorTurns[i];
      if (t.role === "assistant" && t.content.trim().length >= 12) {
        return t.content.trim().slice(0, 800);
      }
    }
    return null;
  }

  if (isImageGenerateRequest(text)) {
    if (pending) return pending;
    for (let i = priorTurns.length - 1; i >= 0; i--) {
      const t = priorTurns[i];
      if (t.role === "assistant" && t.content.trim().length >= 12) {
        return t.content.trim().slice(0, 800);
      }
    }
  }

  return null;
}
