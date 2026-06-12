/** Ingest PDF, Office, text, and images for chat document mode (browser-only). */

import * as mammoth from "mammoth";
import * as XLSX from "xlsx";
import { prepareImageAttachment, type PendingImage } from "./imageAttachments";

export const MAX_DOCUMENT_BYTES = 15 * 1024 * 1024;
export const MAX_PDF_PAGES_TEXT = 50;
export const MAX_PDF_PAGES_VISION = 8;
/** Min chars from PDF text layer before skipping page render. */
export const PDF_TEXT_LAYER_MIN = 80;

export type DocumentKind = "image" | "pdf" | "text" | "docx" | "xlsx" | "unknown";

export type IngestProgress = (message: string) => void;

export type DocumentPayload = {
  kind: DocumentKind;
  label: string;
  mime: string;
  extractedText: string;
  /** JPEG pages for vision model (scanned PDF, images). */
  visionPages: ArrayBuffer[];
  previewUrl: string | null;
};

export const DOCUMENT_ACCEPT =
  "image/jpeg,image/png,image/webp,image/gif,.heic,.heif," +
  "application/pdf," +
  "text/plain,text/markdown,text/csv,.txt,.md,.csv," +
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx," +
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,.xlsx";

const IMAGE_MIME = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);
const HEIC_RE = /\.(heic|heif)$/i;

export const detectDocumentKind = (file: File): DocumentKind => {
  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  if (IMAGE_MIME.has(type) || /\.(jpe?g|png|webp|gif)$/i.test(name)) return "image";
  if (HEIC_RE.test(name) || type === "image/heic" || type === "image/heif") return "image";
  if (type === "application/pdf" || name.endsWith(".pdf")) return "pdf";
  if (
    type.startsWith("text/") ||
    /\.(txt|md|markdown|csv)$/i.test(name)
  ) {
    return "text";
  }
  if (
    type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    name.endsWith(".docx")
  ) {
    return "docx";
  }
  if (
    type === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" ||
    name.endsWith(".xlsx")
  ) {
    return "xlsx";
  }
  return "unknown";
};

export const isAcceptedDocumentFile = (file: File): boolean =>
  detectDocumentKind(file) !== "unknown";

export type PendingAttachment = {
  id: string;
  file: File;
  kind: DocumentKind;
  label: string;
  previewUrl: string | null;
  mime: string;
  extractedText: string;
  visionPages: ArrayBuffer[];
};

const canvasToJpegBytes = async (
  canvas: HTMLCanvasElement,
  maxDim: number,
): Promise<ArrayBuffer> => {
  const sw = canvas.width;
  const sh = canvas.height;
  const scale = Math.min(1, maxDim / Math.max(sw, sh));
  if (scale < 1) {
    const w = Math.max(1, Math.round(sw * scale));
    const h = Math.max(1, Math.round(sh * scale));
    const scaled = document.createElement("canvas");
    scaled.width = w;
    scaled.height = h;
    const ctx = scaled.getContext("2d");
    if (!ctx) throw new Error("Canvas לא נתמך");
    ctx.drawImage(canvas, 0, 0, w, h);
    canvas = scaled;
  }
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("JPEG failed"))), "image/jpeg", 0.88);
  });
  return blob.arrayBuffer();
};

const loadPdfJs = async () => {
  const pdfjs = await import("pdfjs-dist");
  if (typeof window !== "undefined" && !pdfjs.GlobalWorkerOptions.workerSrc) {
    pdfjs.GlobalWorkerOptions.workerSrc = new URL(
      "pdfjs-dist/build/pdf.worker.min.mjs",
      import.meta.url,
    ).toString();
  }
  return pdfjs;
};

const ingestPdf = async (file: File, onProgress?: IngestProgress): Promise<DocumentPayload> => {
  const pdfjs = await loadPdfJs();
  onProgress?.("טוען PDF…");
  const data = new Uint8Array(await file.arrayBuffer());
  const doc = await pdfjs.getDocument({ data }).promise;
  const pageCount = Math.min(doc.numPages, MAX_PDF_PAGES_TEXT);
  const textParts: string[] = [];

  for (let i = 1; i <= pageCount; i++) {
    onProgress?.(`קורא PDF עמוד ${i}/${pageCount}…`);
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    if (pageText) textParts.push(`--- עמוד ${i} ---\n${pageText}`);
  }

  const extractedText = textParts.join("\n\n");
  let fullText = extractedText;
  if (doc.numPages > MAX_PDF_PAGES_TEXT) {
    fullText += `\n\n(… ${doc.numPages - MAX_PDF_PAGES_TEXT} עמודים נוספים לא נכללו)`;
  }

  const visionPages: ArrayBuffer[] = [];
  let previewUrl: string | null = null;

  const needsVision = fullText.length < PDF_TEXT_LAYER_MIN && doc.numPages > 0;
  if (needsVision) {
    const renderCount = Math.min(doc.numPages, MAX_PDF_PAGES_VISION);
    for (let i = 1; i <= renderCount; i++) {
      onProgress?.(`מצייר PDF לניתוח ${i}/${renderCount}…`);
      const page = await doc.getPage(i);
      const viewport = page.getViewport({ scale: 1.5 });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) break;
      await page.render({ canvasContext: ctx, viewport, canvas }).promise;
      const bytes = await canvasToJpegBytes(canvas, 2048);
      visionPages.push(bytes);
      if (i === 1) {
        previewUrl = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
      }
    }
  }

  return {
    kind: "pdf",
    label: file.name,
    mime: "application/pdf",
    extractedText: fullText,
    visionPages,
    previewUrl,
  };
};

const ingestText = async (file: File): Promise<DocumentPayload> => {
  const buf = await file.arrayBuffer();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let text = decoder.decode(buf);
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  return {
    kind: "text",
    label: file.name,
    mime: file.type || "text/plain",
    extractedText: text.trim(),
    visionPages: [],
    previewUrl: null,
  };
};

const ingestDocx = async (file: File, onProgress?: IngestProgress): Promise<DocumentPayload> => {
  onProgress?.("קורא Word…");
  const buf = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer: buf });
  return {
    kind: "docx",
    label: file.name,
    mime: file.type,
    extractedText: result.value.trim(),
    visionPages: [],
    previewUrl: null,
  };
};

const ingestXlsx = async (file: File, onProgress?: IngestProgress): Promise<DocumentPayload> => {
  onProgress?.("קורא Excel…");
  const wb = XLSX.read(await file.arrayBuffer(), { type: "array" });
  const parts: string[] = [];
  for (const sheetName of wb.SheetNames.slice(0, 5)) {
    const csv = XLSX.utils.sheet_to_csv(wb.Sheets[sheetName]).trim();
    if (csv) parts.push(`## ${sheetName}\n${csv}`);
  }
  return {
    kind: "xlsx",
    label: file.name,
    mime: file.type,
    extractedText: parts.join("\n\n"),
    visionPages: [],
    previewUrl: null,
  };
};

const convertHeicToJpegFile = async (file: File): Promise<File> => {
  const heic2any = (await import("heic2any")).default;
  const blob = await heic2any({ blob: file, toType: "image/jpeg", quality: 0.9 });
  const out = Array.isArray(blob) ? blob[0] : blob;
  const name = file.name.replace(/\.(heic|heif)$/i, ".jpg");
  return new File([out], name, { type: "image/jpeg" });
};

const fromPendingImage = (img: PendingImage): PendingAttachment => ({
  id: img.id,
  file: img.file,
  kind: "image",
  label: img.file.name,
  previewUrl: img.previewUrl,
  mime: img.mime,
  extractedText: "",
  visionPages: [img.modelBytes],
});

export const ingestDocument = async (
  file: File,
  onProgress?: IngestProgress,
): Promise<PendingAttachment> => {
  if (file.size > MAX_DOCUMENT_BYTES) {
    throw new Error("הקובץ גדול מדי (מקסימום ~15MB)");
  }

  let workFile = file;
  if (HEIC_RE.test(file.name) || file.type === "image/heic" || file.type === "image/heif") {
    onProgress?.("ממיר HEIC…");
    workFile = await convertHeicToJpegFile(file);
  }

  const kind = detectDocumentKind(workFile);

  if (kind === "image") {
    onProgress?.("מכין תמונה…");
    return fromPendingImage(await prepareImageAttachment(workFile));
  }
  if (kind === "pdf") {
    const payload = await ingestPdf(workFile, onProgress);
    return {
      id: crypto.randomUUID(),
      file,
      kind: payload.kind,
      label: payload.label,
      previewUrl: payload.previewUrl,
      mime: payload.mime,
      extractedText: payload.extractedText,
      visionPages: payload.visionPages,
    };
  }
  if (kind === "text") {
    const payload = await ingestText(workFile);
    return {
      id: crypto.randomUUID(),
      file,
      kind: payload.kind,
      label: payload.label,
      previewUrl: null,
      mime: payload.mime,
      extractedText: payload.extractedText,
      visionPages: [],
    };
  }
  if (kind === "docx") {
    const payload = await ingestDocx(workFile, onProgress);
    return {
      id: crypto.randomUUID(),
      file,
      kind: payload.kind,
      label: payload.label,
      previewUrl: null,
      mime: payload.mime,
      extractedText: payload.extractedText,
      visionPages: [],
    };
  }
  if (kind === "xlsx") {
    const payload = await ingestXlsx(workFile, onProgress);
    return {
      id: crypto.randomUUID(),
      file,
      kind: payload.kind,
      label: payload.label,
      previewUrl: null,
      mime: payload.mime,
      extractedText: payload.extractedText,
      visionPages: [],
    };
  }

  throw new Error("פורמט לא נתמך — PDF, תמונה, TXT, DOCX, XLSX, HEIC");
};

export const buildIngestedDocumentPromptBlock = (attachments: { kind: string; label: string; extractedText: string }[]): string => {
  const blocks = attachments
    .map((a) => {
      const t = a.extractedText.trim();
      if (!t) return "";
      return `[${a.kind.toUpperCase()}: ${a.label}]\n${t}`;
    })
    .filter(Boolean);
  return blocks.join("\n\n");
};

export const hasSubstantialExtractedText = (attachments: { extractedText: string }[]): boolean =>
  attachments.some((a) => a.extractedText.trim().length >= 80);

export const attachmentKindLabel = (kind: DocumentKind): string => {
  const map: Record<DocumentKind, string> = {
    image: "תמונה",
    pdf: "PDF",
    text: "טקסט",
    docx: "Word",
    xlsx: "Excel",
    unknown: "קובץ",
  };
  return map[kind];
};

export const revokePendingAttachment = (p: PendingAttachment) => {
  if (p.previewUrl) URL.revokeObjectURL(p.previewUrl);
};
