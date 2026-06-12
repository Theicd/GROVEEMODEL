/** Lightweight markdown rendering for chat bubbles — no HTML artifact generation. */

import type { ReactNode } from "react";

const escapeHtml = (s: string): string =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const inlineFormat = (raw: string): string =>
  escapeHtml(raw)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, '<code class="chat-md-inline">$1</code>');

export const stripHtmlFencesForChat = (text: string): string => {
  let t = text.trim();
  const htmlFence = t.match(/```html\s*([\s\S]*?)```/i);
  if (htmlFence) {
    const inner = htmlFence[1].trim();
    const body = inner.match(/<body[^>]*>([\s\S]*?)<\/body>/i)?.[1] ?? inner;
    const plain = body
      .replace(/<br\s*\/?>/gi, "\n")
      .replace(/<\/p>/gi, "\n\n")
      .replace(/<\/h[1-6]>/gi, "\n\n")
      .replace(/<\/li>/gi, "\n")
      .replace(/<[^>]+>/g, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    const before = t.slice(0, htmlFence.index).trim();
    return [before, plain].filter(Boolean).join("\n\n");
  }
  return t;
};

export function ChatMarkdown({ text }: { text: string }) {
  const normalized = stripHtmlFencesForChat(text);
  const lines = normalized.replace(/\r\n/g, "\n").split("\n");
  const nodes: ReactNode[] = [];
  let ul: string[] = [];
  let ol: string[] = [];
  let inCode = false;
  let codeLines: string[] = [];
  let key = 0;

  const flushUl = () => {
    if (!ul.length) return;
    nodes.push(
      <ul key={key++} className="chat-md-ul">
        {ul.map((item, i) => (
          <li key={i} dangerouslySetInnerHTML={{ __html: inlineFormat(item) }} />
        ))}
      </ul>,
    );
    ul = [];
  };

  const flushOl = () => {
    if (!ol.length) return;
    nodes.push(
      <ol key={key++} className="chat-md-ol">
        {ol.map((item, i) => (
          <li key={i} dangerouslySetInnerHTML={{ __html: inlineFormat(item) }} />
        ))}
      </ol>,
    );
    ol = [];
  };

  const flushCode = () => {
    if (!inCode) return;
    nodes.push(
      <pre key={key++} className="chat-md-pre">
        <code>{codeLines.join("\n")}</code>
      </pre>,
    );
    codeLines = [];
    inCode = false;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      if (inCode) flushCode();
      else {
        flushUl();
        flushOl();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(raw);
      continue;
    }

    const t = line.trim();
    if (!t) {
      flushUl();
      flushOl();
      continue;
    }

    const h3 = t.match(/^###\s+(.+)/);
    if (h3) {
      flushUl();
      flushOl();
      nodes.push(
        <h3
          key={key++}
          className="chat-md-h3"
          dangerouslySetInnerHTML={{ __html: inlineFormat(h3[1]) }}
        />,
      );
      continue;
    }
    const h2 = t.match(/^##\s+(.+)/);
    if (h2) {
      flushUl();
      flushOl();
      nodes.push(
        <h2
          key={key++}
          className="chat-md-h2"
          dangerouslySetInnerHTML={{ __html: inlineFormat(h2[1]) }}
        />,
      );
      continue;
    }
    const h1 = t.match(/^#\s+(.+)/);
    if (h1) {
      flushUl();
      flushOl();
      nodes.push(
        <h1
          key={key++}
          className="chat-md-h1"
          dangerouslySetInnerHTML={{ __html: inlineFormat(h1[1]) }}
        />,
      );
      continue;
    }
    if (/^[-*•]\s+/.test(t)) {
      flushOl();
      ul.push(t.replace(/^[-*•]\s+/, ""));
      continue;
    }
    const num = t.match(/^\d+[.)]\s+(.+)/);
    if (num) {
      flushUl();
      ol.push(`${t.match(/^\d+[.)]/)?.[0] ?? ""} ${num[1]}`.trim());
      continue;
    }
    if (t.startsWith(">")) {
      flushUl();
      flushOl();
      nodes.push(
        <blockquote
          key={key++}
          className="chat-md-quote"
          dangerouslySetInnerHTML={{ __html: inlineFormat(t.replace(/^>\s?/, "")) }}
        />,
      );
      continue;
    }

    flushUl();
    flushOl();
    nodes.push(
      <p key={key++} className="chat-md-p" dangerouslySetInnerHTML={{ __html: inlineFormat(t) }} />,
    );
  }

  flushUl();
  flushOl();
  flushCode();

  if (!nodes.length) {
    return <p className="chat-md-p">{normalized}</p>;
  }

  return <div className="chat-md">{nodes}</div>;
}
