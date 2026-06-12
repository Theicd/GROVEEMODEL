import { describe, expect, it } from "vitest";
import { detectDocumentKind, isAcceptedDocumentFile } from "./documentIngest";

describe("documentIngest", () => {
  it("accepts common document types", () => {
    expect(isAcceptedDocumentFile({ type: "application/pdf", name: "hw.pdf" } as File)).toBe(true);
    expect(isAcceptedDocumentFile({ type: "text/plain", name: "notes.txt" } as File)).toBe(true);
    expect(isAcceptedDocumentFile({
      type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      name: "doc.docx",
    } as File)).toBe(true);
    expect(isAcceptedDocumentFile({
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      name: "sheet.xlsx",
    } as File)).toBe(true);
    expect(isAcceptedDocumentFile({ type: "image/png", name: "a.png" } as File)).toBe(true);
    expect(isAcceptedDocumentFile({ type: "", name: "photo.heic" } as File)).toBe(true);
    expect(isAcceptedDocumentFile({ type: "application/zip", name: "x.zip" } as File)).toBe(false);
  });

  it("detects document kind from mime and extension", () => {
    expect(detectDocumentKind({ type: "application/pdf", name: "a.pdf" } as File)).toBe("pdf");
    expect(detectDocumentKind({ type: "", name: "a.md" } as File)).toBe("text");
    expect(detectDocumentKind({ type: "", name: "a.docx" } as File)).toBe("docx");
  });
});
