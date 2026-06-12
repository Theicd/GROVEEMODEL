/** Rich HTML canvas for document / image text answers — RTL, headings, dynamic background. */

import { wantsWorksheetReplicaHtml } from "./chatIntents";

const isWorksheetReplicaPage = (html: string): boolean =>
  /worksheet-replica|class="ws-|ws-table|ws-answer|@media print/i.test(html);

export type DocumentCanvasParams = {
  title: string;
  bodyText: string;
  subtitle?: string;
  badge?: string;
};

const escapeHtml = (s: string): string =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

/** Lightweight markdown → HTML for chat answers (headings, lists, quotes, code). */
export const markdownToDocumentBodyHtml = (text: string): string => {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];
  let inUl = false;
  let inOl = false;
  let inPre = false;
  let preBuf: string[] = [];

  const closeLists = () => {
    if (inUl) {
      out.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      out.push("</ol>");
      inOl = false;
    }
  };

  const flushPre = () => {
    if (!inPre) return;
    out.push(`<pre class="doc-code"><code>${escapeHtml(preBuf.join("\n"))}</code></pre>`);
    preBuf = [];
    inPre = false;
  };

  const inlineFmt = (s: string) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code class=\"doc-inline\">$1</code>");

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      if (inPre) flushPre();
      else {
        closeLists();
        inPre = true;
      }
      continue;
    }
    if (inPre) {
      preBuf.push(raw);
      continue;
    }

    const t = line.trim();
    if (!t) {
      closeLists();
      continue;
    }

    const h3 = t.match(/^###\s+(.+)/);
    if (h3) {
      closeLists();
      out.push(`<h3 class="doc-h3">${inlineFmt(h3[1])}</h3>`);
      continue;
    }
    const h2 = t.match(/^##\s+(.+)/);
    if (h2) {
      closeLists();
      out.push(`<h2 class="doc-h2">${inlineFmt(h2[1])}</h2>`);
      continue;
    }
    const h1 = t.match(/^#\s+(.+)/);
    if (h1) {
      closeLists();
      out.push(`<h1 class="doc-h1">${inlineFmt(h1[1])}</h1>`);
      continue;
    }
    if (/^[-*•]\s+/.test(t)) {
      if (!inUl) {
        closeLists();
        out.push('<ul class="doc-ul">');
        inUl = true;
      }
      out.push(`<li>${inlineFmt(t.replace(/^[-*•]\s+/, ""))}</li>`);
      continue;
    }
    const num = t.match(/^\d+[.)]\s+(.+)/);
    if (num) {
      if (!inOl) {
        closeLists();
        out.push('<ol class="doc-ol">');
        inOl = true;
      }
      out.push(`<li>${inlineFmt(num[1])}</li>`);
      continue;
    }
    if (t.startsWith(">")) {
      closeLists();
      out.push(`<blockquote class="doc-quote">${inlineFmt(t.replace(/^>\s?/, ""))}</blockquote>`);
      continue;
    }
    closeLists();
    out.push(`<p class="doc-p">${inlineFmt(t)}</p>`);
  }

  flushPre();
  closeLists();
  return out.join("\n");
};

const CANVAS_STYLES = `
:root {
  --bg1: #0a0e1a;
  --bg2: #121a32;
  --accent: #3dffb8;
  --accent2: #5b8cff;
  --text: #e8eeff;
  --muted: #9aa8c9;
  --card: rgba(18, 26, 48, 0.72);
  --border: rgba(93, 140, 255, 0.28);
}
*, *::before, *::after { box-sizing: border-box; }
html, body { min-height: 100%; margin: 0; }
body {
  font-family: "Segoe UI", "Rubik", "Arial Hebrew", system-ui, sans-serif;
  color: var(--text);
  background: var(--bg1);
  line-height: 1.65;
  overflow-x: hidden;
}
.bg-layer {
  position: fixed; inset: 0; z-index: 0; overflow: hidden;
  background: radial-gradient(ellipse 120% 80% at 20% 10%, rgba(61,255,184,0.12), transparent 55%),
              radial-gradient(ellipse 90% 70% at 85% 90%, rgba(91,140,255,0.18), transparent 50%),
              linear-gradient(145deg, var(--bg1), var(--bg2));
}
.bg-orb {
  position: absolute; border-radius: 50%; filter: blur(60px); opacity: 0.45;
  animation: drift 18s ease-in-out infinite alternate;
}
.bg-orb.a { width: 280px; height: 280px; background: #3dffb8; top: 8%; left: 5%; animation-delay: 0s; }
.bg-orb.b { width: 340px; height: 340px; background: #5b8cff; bottom: 10%; right: 8%; animation-delay: -6s; }
.bg-orb.c { width: 200px; height: 200px; background: #a855f7; top: 45%; left: 55%; animation-delay: -12s; opacity: 0.25; }
@keyframes drift {
  from { transform: translate(0, 0) scale(1); }
  to { transform: translate(24px, -18px) scale(1.08); }
}
.grid-overlay {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.06;
  background-image: linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px);
  background-size: 32px 32px;
}
.doc-shell {
  position: relative; z-index: 1; max-width: 720px; margin: 0 auto;
  padding: 28px 20px 48px;
  animation: fadeUp 0.55s ease-out both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
.doc-card {
  background: var(--card);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 28px 26px 32px;
  box-shadow: 0 24px 64px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
}
.doc-badge {
  display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--accent); background: rgba(61,255,184,0.12);
  border: 1px solid rgba(61,255,184,0.35); border-radius: 999px; padding: 4px 12px; margin-bottom: 14px;
}
.doc-title {
  margin: 0 0 8px; font-size: clamp(1.35rem, 4vw, 1.85rem); font-weight: 800;
  background: linear-gradient(120deg, #fff 30%, var(--accent) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.doc-subtitle { margin: 0 0 22px; color: var(--muted); font-size: 0.95rem; }
.doc-body { font-size: 1rem; }
.doc-h1 { font-size: 1.35rem; margin: 1.4em 0 0.5em; color: #fff; border-bottom: 1px solid var(--border); padding-bottom: 0.35em; }
.doc-h2 { font-size: 1.15rem; margin: 1.2em 0 0.45em; color: var(--accent); }
.doc-h3 { font-size: 1.02rem; margin: 1em 0 0.35em; color: #c5d4ff; }
.doc-p { margin: 0.65em 0; color: var(--text); }
.doc-ul, .doc-ol { margin: 0.6em 0 0.8em; padding-inline-start: 1.35em; }
.doc-ul li, .doc-ol li { margin: 0.35em 0; }
.doc-ul li::marker { color: var(--accent); }
.doc-quote {
  margin: 1em 0; padding: 12px 16px; border-inline-start: 3px solid var(--accent2);
  background: rgba(91,140,255,0.1); border-radius: 0 12px 12px 0; color: #d0dcff; font-style: italic;
}
.doc-code {
  margin: 1em 0; padding: 14px 16px; background: rgba(0,0,0,0.45);
  border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
  overflow-x: auto; font-size: 0.88rem; line-height: 1.5; direction: ltr; text-align: left;
}
.doc-inline {
  background: rgba(61,255,184,0.12); color: var(--accent); padding: 0.1em 0.35em;
  border-radius: 4px; font-size: 0.92em;
}
.doc-footer {
  margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);
  font-size: 0.78rem; color: var(--muted); text-align: center;
}
`;

export const buildDocumentCanvasHtml = (params: DocumentCanvasParams): string => {
  const title = escapeHtml(params.title || "תוכן המסמך");
  const subtitle = params.subtitle ? `<p class="doc-subtitle">${escapeHtml(params.subtitle)}</p>` : "";
  const badge = params.badge
    ? `<span class="doc-badge">${escapeHtml(params.badge)}</span>`
    : `<span class="doc-badge">GROVEE · Canvas</span>`;
  const bodyHtml = markdownToDocumentBodyHtml(params.bodyText);

  return `<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title>
<style>${CANVAS_STYLES}</style>
</head>
<body>
<div class="bg-layer" aria-hidden="true">
  <div class="bg-orb a"></div>
  <div class="bg-orb b"></div>
  <div class="bg-orb c"></div>
</div>
<div class="grid-overlay" aria-hidden="true"></div>
<main class="doc-shell">
  <article class="doc-card">
    ${badge}
    <h1 class="doc-title">${title}</h1>
    ${subtitle}
    <div class="doc-body">${bodyHtml}</div>
    <footer class="doc-footer">נוצר ב-GROVEE Studio · תצוגת מסמך</footer>
  </article>
</main>
</body>
</html>`;
};

/** True when model HTML is bare/plain (white page, no real layout). */
export const isMinimalDocumentHtml = (html: string): boolean => {
  if (isWorksheetReplicaPage(html)) return false;
  const t = html.trim().toLowerCase();
  if (t.length < 40) return true;
  const hasRichLayout =
    /gradient|animation|backdrop-filter|glass|doc-card|doc-shell|bg-layer|@keyframes/i.test(html);
  if (hasRichLayout) return false;
  const styleLen = (html.match(/<style[^>]*>([\s\S]*?)<\/style>/gi) ?? [])
    .join("")
    .length;
  if (styleLen < 120) return true;
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  const body = bodyMatch?.[1] ?? html;
  const plain = body.replace(/<[^>]+>/g, "").trim();
  if (plain.startsWith("```") || !/<h[1-6]|<ul|<ol|<blockquote|<article/i.test(body)) {
    return plain.length > 30;
  }
  return false;
};

const stripMarkdownFences = (text: string): string =>
  text.replace(/^```[\w]*\s*/gm, "").replace(/```\s*$/gm, "").trim();

/** Re-wrap weak model HTML into the studio canvas template. */
export const wrapMinimalHtmlInCanvas = (html: string, title: string): string => {
  const bodyText = extractPlainBodyFromHtml(html) || stripMarkdownFences(html);
  return buildDocumentCanvasHtml({ title, bodyText, subtitle: "תמלול וניתוח מהתמונה" });
};

export const extractPlainBodyFromHtml = (html: string): string => {
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  const inner = bodyMatch?.[1] ?? html;
  return stripMarkdownFences(inner.replace(/<[^>]+>/g, "\n").replace(/\n{3,}/g, "\n\n")).trim();
};

export const wantsDocumentCanvasDisplay = (text: string): boolean => {
  const t = text.trim();
  if (!t) return true;
  if (wantsWorksheetReplicaHtml(t)) return false;
  return (
    /מה כתוב|מה רשום|what('s| is) written|what does (it|the image) say/i.test(t) ||
    /תסתכל על (ה)?תמונה|על התמונה|look at (the )?(image|picture)/i.test(t) ||
    /שיעורי\s*(ה)?בית|homework|שאל(ות|ה)|worksheet|מסמך|document/i.test(t) ||
    /הצג|תצוגה|סדר|מסודר|canvas|html/i.test(t)
  );
};

export const documentCanvasTitleFromPrompt = (prompt: string): string => {
  const t = prompt.trim();
  if (/מה כתוב|מה רשום/i.test(t)) return "מה כתוב בתמונה";
  if (/שיעורי\s*(ה)?בית|homework/i.test(t)) return "שיעורי בית";
  if (/שאל/i.test(t)) return "שאלות מהמסמך";
  if (t.length > 4 && t.length <= 48) return t;
  return "תוכן המסמך";
};
