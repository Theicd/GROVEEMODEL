/** Fillable worksheet HTML — white A4 page replica for print / re-photo. */

import { markdownToDocumentBodyHtml } from "./documentHtmlCanvas";

export type WorksheetReplicaParams = {
  title?: string;
  /** Inner body HTML or plain text (converted to simple layout). */
  body: string;
  rtl?: boolean;
};

const WORKSHEET_STYLES = `
:root {
  --page-w: 210mm;
  --ink: #111;
  --line: #333;
  --muted: #555;
}
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "Times New Roman", "David", "Arial Hebrew", Georgia, serif;
  font-size: 14pt;
  line-height: 1.45;
  color: var(--ink);
  background: #e8e8e8;
}
@media screen {
  body { padding: 16px 12px 32px; }
}
.page {
  width: 100%;
  max-width: var(--page-w);
  min-height: 297mm;
  margin: 0 auto;
  padding: 18mm 16mm 22mm;
  background: #fff;
  box-shadow: 0 2px 24px rgba(0,0,0,0.12);
}
@media print {
  body { background: #fff; padding: 0; }
  .page { box-shadow: none; max-width: none; width: auto; min-height: auto; margin: 0; padding: 12mm; }
  .no-print { display: none !important; }
  input, textarea { border-bottom: 1px solid #000 !important; background: transparent !important; }
}
.ws-header { text-align: center; margin-bottom: 1.2em; border-bottom: 2px solid var(--line); padding-bottom: 0.6em; }
.ws-title { font-size: 1.25em; font-weight: 700; margin: 0 0 0.25em; }
.ws-sub { font-size: 0.85em; color: var(--muted); margin: 0; }
.ws-q {
  margin: 1em 0 0.6em;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.35em 0.5em;
}
.ws-q-num { font-weight: 700; min-width: 1.6em; }
.ws-q-text { flex: 1 1 60%; }
.ws-answer {
  display: block;
  width: 100%;
  margin: 0.35em 0 0.8em;
  padding: 0.35em 0.5em;
  border: none;
  border-bottom: 1.5px solid var(--line);
  background: #fafafa;
  font: inherit;
  color: var(--ink);
}
textarea.ws-answer {
  min-height: 3.2em;
  border: 1px solid #ccc;
  border-radius: 2px;
  resize: vertical;
}
.ws-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.8em 0 1em;
  font-size: 0.95em;
}
.ws-table th, .ws-table td {
  border: 1px solid var(--line);
  padding: 0.45em 0.55em;
  vertical-align: top;
}
.ws-table th { background: #f5f5f5; font-weight: 700; }
.ws-table input.ws-answer {
  margin: 0;
  border-bottom: 1px solid #999;
  background: transparent;
}
.ws-blank {
  display: inline-block;
  min-width: 4em;
  border-bottom: 1.5px solid var(--line);
  height: 1.2em;
  vertical-align: bottom;
}
.ws-footer {
  margin-top: 2em;
  padding-top: 0.6em;
  border-top: 1px solid #ccc;
  font-size: 0.75em;
  color: var(--muted);
  text-align: center;
}
.no-print {
  text-align: center;
  margin-bottom: 12px;
  font-family: system-ui, sans-serif;
  font-size: 13px;
  color: #666;
}
`;

const escapeHtml = (s: string): string =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Build fillable worksheet from plain/OCR text (numbered lines → inputs). */
export const buildSimpleWorksheetFromPlainText = (text: string, title = "דף עבודה"): string => {
  const lines = text.replace(/\r\n/g, "\n").split("\n").map((l) => l.trim()).filter(Boolean);
  const blocks: string[] = [];

  for (const line of lines) {
    const q = line.match(/^(\d+[.)]\s*)(.+)/);
    if (q) {
      blocks.push(
        `<div class="ws-q"><span class="ws-q-num">${escapeHtml(q[1])}</span><span class="ws-q-text">${escapeHtml(q[2])}</span></div>`,
        `<input class="ws-answer" type="text" aria-label="תשובה לשאלה ${escapeHtml(q[1])}" />`,
      );
      continue;
    }
    if (/^[-*•]\s+/.test(line)) {
      blocks.push(`<p>${escapeHtml(line.replace(/^[-*•]\s+/, "• "))}</p>`);
      continue;
    }
    blocks.push(`<p>${escapeHtml(line)}</p>`);
  }

  return buildWorksheetReplicaHtml({ title, body: blocks.join("\n"), rtl: true });
};

export const buildWorksheetReplicaHtml = (params: WorksheetReplicaParams): string => {
  const rtl = params.rtl !== false;
  const title = escapeHtml(params.title || "דף עבודה");
  const body = params.body.trim().startsWith("<")
    ? params.body
    : markdownToDocumentBodyHtml(params.body)
        .replace(/<p class="doc-p">/g, '<p class="ws-p">')
        .replace(/class="doc-h1"/g, 'class="ws-title"')
        .replace(/class="doc-h2"/g, 'class="ws-section"');

  return `<!DOCTYPE html>
<html lang="${rtl ? "he" : "en"}" dir="${rtl ? "rtl" : "ltr"}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title>
<style>${WORKSHEET_STYLES}
.ws-p { margin: 0.5em 0; }
.ws-section { font-size: 1.05em; font-weight: 700; margin: 1em 0 0.5em; }
</style>
</head>
<body>
<p class="no-print">מלא תשובות → הדפס (Ctrl+P) או צלם מחדש</p>
<div class="page worksheet-replica">
  <header class="ws-header">
    <h1 class="ws-title">${title}</h1>
    <p class="ws-sub">דף עבודה למילוי</p>
  </header>
  <main class="ws-body">${body}</main>
  <footer class="ws-footer">GROVEE · דף עבודה להדפסה</footer>
</div>
</body>
</html>`;
};

export const isWorksheetReplicaArtifact = (html: string): boolean =>
  /worksheet-replica|class="ws-|ws-table|ws-answer|@media print/i.test(html);

/** Wrap fragment or weak HTML in the A4 worksheet shell. */
export const enhanceWorksheetReplicaHtml = (html: string, title = "דף עבודה"): string => {
  const t = html.trim();
  if (isWorksheetReplicaArtifact(t) && /<style/i.test(t) && t.length > 400) return t;

  const fullDoc = t.match(/<html[\s\S]*<\/html>/i);
  if (fullDoc && isWorksheetReplicaArtifact(fullDoc[0])) return fullDoc[0];

  const bodyMatch = t.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  const inner = bodyMatch?.[1]?.trim() ?? t.replace(/^```html\s*/i, "").replace(/```\s*$/, "").trim();

  if (inner.startsWith("<!DOCTYPE") || inner.startsWith("<html")) {
    return buildWorksheetReplicaHtml({ title, body: extractPlainInner(inner), rtl: true });
  }

  return buildWorksheetReplicaHtml({ title, body: inner, rtl: true });
};

const extractPlainInner = (html: string): string => {
  const body = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i)?.[1] ?? html;
  return body.trim();
};

export const worksheetTitleFromPrompt = (prompt: string): string => {
  const t = prompt.trim();
  if (/דף עבודה|worksheet/i.test(t)) return "דף עבודה";
  if (/שיעורי\s*(ה)?בית|homework/i.test(t)) return "שיעורי בית";
  if (/מבחן|test/i.test(t)) return "דף מבחן";
  return "דף עבודה";
};
