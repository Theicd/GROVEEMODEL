import { describe, expect, it } from "vitest";
import {
  buildDocumentCanvasHtml,
  isMinimalDocumentHtml,
  markdownToDocumentBodyHtml,
  wrapMinimalHtmlInCanvas,
} from "./documentHtmlCanvas";

describe("documentHtmlCanvas", () => {
  it("builds rich HTML with gradient layout", () => {
    const html = buildDocumentCanvasHtml({
      title: "מה כתוב בתמונה",
      bodyText: "## כותרת\n\n- שורה אחת\n- שורה שנייה",
    });
    expect(html).toContain("bg-layer");
    expect(html).toContain("doc-card");
    expect(html).toContain("@keyframes");
    expect(html).toContain("כותרת");
  });

  it("parses markdown sections", () => {
    const body = markdownToDocumentBodyHtml("## סעיף\n\nטקסט רגיל\n\n- נקודה");
    expect(body).toContain('<h2 class="doc-h2">');
    expect(body).toContain("<ul");
    expect(body).toContain("<li>");
  });

  it("detects minimal HTML and wraps it", () => {
    const plain = `<!DOCTYPE html><html><body>\`\`\`\nGROVEE STUDIO\n\`\`\`</body></html>`;
    expect(isMinimalDocumentHtml(plain)).toBe(true);
    const wrapped = wrapMinimalHtmlInCanvas(plain, "מה כתוב");
    expect(wrapped).toContain("doc-card");
    expect(wrapped).toContain("GROVEE STUDIO");
  });
});
