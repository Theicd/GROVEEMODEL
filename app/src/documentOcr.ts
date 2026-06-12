/** OCR for document photos — Hebrew, English, and mixed text via Tesseract.js (lazy-loaded). */

let tesseractModule: typeof import("tesseract.js") | null = null;

const loadTesseract = async (): Promise<typeof import("tesseract.js")> => {
  if (!tesseractModule) {
    tesseractModule = await import("tesseract.js");
  }
  return tesseractModule;
};

export type OcrProgress = (message: string) => void;

/** Extract text from one or more JPEG/PNG image buffers. Returns empty string if OCR fails or finds nothing. */
export const extractTextFromDocumentImages = async (
  buffers: ArrayBuffer[],
  onProgress?: OcrProgress,
): Promise<string> => {
  if (!buffers.length) return "";

  const Tesseract = await loadTesseract();
  const parts: string[] = [];

  for (let i = 0; i < buffers.length; i++) {
    onProgress?.(`קורא טקסט מתמונה ${i + 1}/${buffers.length}…`);
    const blob = new Blob([buffers[i]], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    try {
      const result = await Tesseract.recognize(url, "heb+eng", {
        logger: (m) => {
          if (m.status === "recognizing text" && m.progress > 0 && m.progress < 1) {
            onProgress?.(`OCR ${Math.round(m.progress * 100)}%…`);
          }
        },
      });
      const text = result.data.text.replace(/\s+\n/g, "\n").trim();
      if (text) {
        parts.push(buffers.length > 1 ? `--- תמונה ${i + 1} ---\n${text}` : text);
      }
    } catch {
      /* vision model may still read the image */
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  return parts.join("\n\n");
};
