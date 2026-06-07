import {
  findAnswerContentStart,
  getArtifactScanContent,
  splitAssistantStream,
  stripGemmaControlTokens,
  cleanDisplayText,
} from "./chatIntents";
import type { Artifact } from "./ArtifactPanel";

type MsgPart = { type: "text" | "html" | "image" | "code"; value: string; lang?: string };

const extractRichParts = (content: string): MsgPart[] => {
  const parts: MsgPart[] = [];
  let remaining = content;
  const pushText = (t: string) => {
    if (t) parts.push({ type: "text", value: t });
  };

  while (remaining.length) {
    type Cand = { idx: number; len: number; part: MsgPart };
    const candidates: Cand[] = [];

    const htmlFence = remaining.match(/```html\s*([\s\S]*?)(?:```|$)/i);
    if (htmlFence && htmlFence.index !== undefined && (htmlFence[1].trim().length > 0 || htmlFence[0].includes("```"))) {
      candidates.push({
        idx: htmlFence.index,
        len: htmlFence[0].length,
        part: { type: "html", value: htmlFence[1].trim() },
      });
    }

    const codeFence = remaining.match(/```(?!html)(\w*)\s*([\s\S]*?)```/i);
    if (codeFence && codeFence.index !== undefined) {
      candidates.push({
        idx: codeFence.index,
        len: codeFence[0].length,
        part: { type: "code", value: codeFence[2], lang: codeFence[1] || "text" },
      });
    }

    const fullDoc = remaining.match(/(?:<!DOCTYPE\s+html[^>]*>|<html\b[^>]*>)[\s\S]*?<\/html>/i);
    if (fullDoc && fullDoc.index !== undefined) {
      candidates.push({
        idx: fullDoc.index,
        len: fullDoc[0].length,
        part: { type: "html", value: fullDoc[0].trim() },
      });
    }

    let best: Cand | null = null;
    for (const c of candidates) {
      if (!best || c.idx < best.idx) best = c;
    }

    if (!best) {
      pushText(remaining);
      break;
    }

    pushText(remaining.slice(0, best.idx));
    parts.push(best.part);
    remaining = remaining.slice(best.idx + best.len);
  }

  if (!parts.length) parts.push({ type: "text", value: content });
  return parts;
};

/** Extract HTML/code artifact from model output (preserves raw source — no line mangling). */
export const extractPrimaryArtifact = (content: string): Artifact | null => {
  const parts = extractRichParts(content);
  const html = parts.find((p) => p.type === "html" && p.value.length > 0);
  if (html) return { kind: "html", content: html.value, title: "HTML" };
  const code = parts.find((p) => p.type === "code" && p.value.length > 0);
  if (code) return { kind: "code", content: code.value, lang: code.lang, title: code.lang || "code" };

  const streamingHtml = content.match(/```html\s*([\s\S]*)$/i);
  if (streamingHtml && streamingHtml[1].trim().length > 8) {
    return { kind: "html", content: streamingHtml[1].trim(), title: "HTML" };
  }
  const streamingCode = content.match(/```(?!html)(\w*)\s*([\s\S]*)$/i);
  if (streamingCode && streamingCode[2].trim().length > 0) {
    return {
      kind: "code",
      content: streamingCode[2],
      lang: streamingCode[1] || "text",
      title: streamingCode[1] || "code",
    };
  }
  return null;
};

/** Plain-text chat line — must NOT mangle HTML/JS (no dedupe, no empty-line removal). */
export const cleanModelOutputForText = (input: string, thinkingEnabled: boolean): string => {
  const scan = getArtifactScanContent(input, thinkingEnabled);
  const base = stripGemmaControlTokens(scan || input);
  const trimmed = base.trim();
  return trimmed.length ? trimmed : "No response generated.";
};

export type PersistedAssistantPayload = {
  content: string;
  artifact: Artifact | null;
  /** Thinking channel — shown in ThinkingBlock, not as plain chat paragraphs. */
  thought?: string;
};

/**
 * Build what we save after generation completes.
 * Code/HTML is stored verbatim in `artifact`; thought stays in `thought` for chat UI.
 */
export const buildPersistedAssistantPayload = (
  raw: string,
  thinkingEnabled: boolean,
): PersistedAssistantPayload => {
  const stream = splitAssistantStream(raw, thinkingEnabled);
  const scanContent = getArtifactScanContent(raw, thinkingEnabled).trim() || stream.answer.trim() || raw.trim();
  const artifact = extractPrimaryArtifact(scanContent);
  const cleanThought = stream.thought ? cleanDisplayText(stream.thought) : undefined;

  if (artifact) {
    const fenceAt = findAnswerContentStart(scanContent);
    const answerIntro = fenceAt > 0 ? scanContent.slice(0, fenceAt).trim() : "";
    const cleanIntro = answerIntro ? cleanDisplayText(answerIntro) : "";
    const content =
      cleanIntro ||
      (artifact.kind === "html"
        ? "יצרתי דף HTML — לחץ «פתח HTML בחלונית»."
        : `יצרתי קוד — לחץ «פתח ${artifact.title} בחלונית».`);
    return { content, artifact, thought: cleanThought || undefined };
  }

  if (cleanThought) {
    const cleanAnswer = cleanDisplayText(scanContent || stream.answer || raw);
    return { content: cleanAnswer || "No response generated.", artifact: null, thought: cleanThought };
  }

  return { content: cleanModelOutputForText(raw, thinkingEnabled), artifact: null };
};

export { extractRichParts };
