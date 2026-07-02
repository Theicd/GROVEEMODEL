/**
 * Document ingest + OCR for small text models (no vision LLM).
 */

import type { PendingAttachment } from "./documentIngest";
import {
  attachmentKindLabel,
  hasSubstantialExtractedText,
  buildIngestedDocumentPromptBlock,
} from "./documentIngest";
import { extractTextFromDocumentImages } from "./documentOcr";
import { wantsExactTextExtraction, wantsWorksheetReplicaHtml } from "./chatIntents";

export type DocumentPipelineResult = {
  contextBlock: string;
  ocrRan: boolean;
  worksheetReplica: boolean;
};

export function buildDocumentContextForSmallModel(text: string, maxChars = 600): string {
  const t = text.trim();
  if (!t) return "";
  if (t.length <= maxChars) return t;
  return `${t.slice(0, maxChars - 1)}…`;
}

export async function runDocumentTurnPipeline(params: {
  trimmed: string;
  attachments: PendingAttachment[];
  onStatus?: (msg: string) => void;
  cache?: Map<string, string>;
}): Promise<DocumentPipelineResult> {
  const { trimmed, attachments, onStatus, cache } = params;

  if (wantsWorksheetReplicaHtml(trimmed)) {
    return { contextBlock: "", ocrRan: false, worksheetReplica: true };
  }

  const ingestedBlock = buildIngestedDocumentPromptBlock(
    attachments.map((p) => ({
      kind: attachmentKindLabel(p.kind),
      label: p.label,
      extractedText: p.extractedText,
    })),
  );

  const attachmentBuffers = attachments.flatMap((p) => p.visionPages);
  const cacheKey = attachments.map((p) => p.id).join("|");
  const cached = cache?.get(cacheKey);

  let ocrText = cached ?? "";
  let ocrRan = false;

  const skipOcr =
    !!cached ||
    hasSubstantialExtractedText(attachments) ||
    (attachmentBuffers.length === 0 && ingestedBlock.length > 0);

  if (!skipOcr && attachmentBuffers.length && wantsExactTextExtraction(trimmed)) {
    onStatus?.("קורא טקסט מהמסמך…");
    try {
      ocrText = await extractTextFromDocumentImages(attachmentBuffers, onStatus);
      ocrRan = true;
      if (ocrText.trim() && cache) cache.set(cacheKey, ocrText);
    } catch {
      ocrText = "";
    }
  }

  const parts: string[] = [];
  if (ingestedBlock.trim()) parts.push(ingestedBlock.trim());
  if (ocrText.trim()) parts.push(ocrText.trim());

  const combined = parts.join("\n\n");
  const contextBlock = combined
    ? `Document content (trust this over guesswork):\n${buildDocumentContextForSmallModel(combined)}`
    : "";

  return { contextBlock, ocrRan, worksheetReplica: false };
}
