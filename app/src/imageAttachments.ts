/** Image attachment helpers for vision chat (compress, validate, read). */

export const MAX_ATTACHMENTS = 4;
export const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
export const MAX_IMAGE_DIMENSION = 2048;
export const THUMB_MAX = 320;

export type PendingImage = {
  id: string;
  file: File;
  previewUrl: string;
  /** Full-resolution bytes for the model (JPEG). */
  modelBytes: ArrayBuffer;
  mime: string;
};

export type StoredMessageImage = {
  id: string;
  previewUrl: string;
};

const ACCEPTED_MIME = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);

export const isAcceptedImageFile = (file: File): boolean =>
  ACCEPTED_MIME.has(file.type) || /\.(jpe?g|png|webp|gif)$/i.test(file.name);

export const defaultVisionPrompt = (rtl: boolean): string =>
  rtl
    ? "מה מוצג בתמונה? תאר בפירוט בעברית."
    : "Describe this image in detail.";

const loadImageElement = (src: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("לא ניתן לטעון את התמונה"));
    img.src = src;
  });

const canvasToBlob = (canvas: HTMLCanvasElement, type: string, quality: number): Promise<Blob> =>
  new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("שמירת תמונה נכשלה"))),
      type,
      quality,
    );
  });

const drawScaled = async (
  source: CanvasImageSource,
  sw: number,
  sh: number,
  maxDim: number,
): Promise<{ blob: Blob; previewUrl: string }> => {
  const scale = Math.min(1, maxDim / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas לא נתמך");
  ctx.drawImage(source, 0, 0, w, h);
  const blob = await canvasToBlob(canvas, "image/jpeg", maxDim <= THUMB_MAX ? 0.82 : 0.88);
  const previewUrl = URL.createObjectURL(blob);
  return { blob, previewUrl };
};

/** Prepare a user-selected file for model + UI preview. */
export const prepareImageAttachment = async (file: File): Promise<PendingImage> => {
  if (!isAcceptedImageFile(file)) {
    throw new Error("פורמט לא נתמך — השתמש ב-JPEG, PNG, WebP או GIF");
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error("התמונה גדולה מדי (מקסימום ~12MB)");
  }

  const objectUrl = URL.createObjectURL(file);
  try {
    const img = await loadImageElement(objectUrl);
    const { blob: modelBlob, previewUrl } = await drawScaled(img, img.naturalWidth, img.naturalHeight, MAX_IMAGE_DIMENSION);
    const modelBytes = await modelBlob.arrayBuffer();
    return {
      id: crypto.randomUUID(),
      file,
      previewUrl,
      modelBytes,
      mime: "image/jpeg",
    };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
};

export const revokePendingImages = (items: PendingImage[]) => {
  for (const p of items) URL.revokeObjectURL(p.previewUrl);
};

export const revokeStoredPreviews = (items: StoredMessageImage[] | undefined) => {
  if (!items) return;
  for (const p of items) URL.revokeObjectURL(p.previewUrl);
};
