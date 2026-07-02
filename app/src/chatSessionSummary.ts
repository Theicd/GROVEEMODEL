/**
 * Browser-only rolling session summary (JARVIS tier-2 equivalent, no server).
 * Older turns compress into a short block injected into the system prompt.
 */

import { collectSessionMemoryFacts } from "./chatSessionMemory";

export type SummaryMessage = { role: string; content: string };

export const SESSION_SUMMARY_MAX_CHARS = 1400;
/** Keep this many recent messages in full before rolling older turns into summary. */
export const SESSION_SUMMARY_KEEP_RECENT = 16;
/** Update summary every N completed turns once over the keep window. */
export const SESSION_SUMMARY_UPDATE_EVERY = 4;

export function formatSessionSummaryForPrompt(summary: string): string {
  const t = summary.trim();
  if (!t) return "";
  return (
    "Earlier in this chat (before the recent messages below):\n" +
    `${t}\n` +
    "Trust recent messages first; use this block for older context and follow-ups."
  );
}

function clipSummary(text: string, max = SESSION_SUMMARY_MAX_CHARS): string {
  const t = text.trim();
  if (t.length <= max) return t;
  return `…${t.slice(t.length - max + 1)}`;
}

/** Merge rotated-out turns + facts into an updated rolling summary (no LLM). */
export function updateRollingSessionSummary(
  currentSummary: string,
  allMessages: SummaryMessage[],
  options: {
    keepRecent?: number;
    maxChars?: number;
  } = {},
): string {
  const keepRecent = options.keepRecent ?? SESSION_SUMMARY_KEEP_RECENT;
  const maxChars = options.maxChars ?? SESSION_SUMMARY_MAX_CHARS;

  const chatRows = allMessages.filter((m) => m.role === "user" || m.role === "assistant");
  if (chatRows.length <= keepRecent) {
    return clipSummary(currentSummary, maxChars);
  }

  const rotateCount = chatRows.length - keepRecent;
  const rotated = chatRows.slice(0, rotateCount);
  const facts = collectSessionMemoryFacts(rotated);

  const lines: string[] = [];
  if (currentSummary.trim()) lines.push(currentSummary.trim());
  if (facts.length) {
    lines.push(`Topics: ${facts.slice(-6).join(" · ")}`);
  }

  for (const m of rotated.slice(-8)) {
    const prefix = m.role === "user" ? "User" : "Assistant";
    const snippet = m.content.replace(/\s+/g, " ").trim().slice(0, 100);
    if (snippet) lines.push(`${prefix}: ${snippet}`);
  }

  return clipSummary(lines.join("\n"), maxChars);
}

export function shouldRefreshSessionSummary(messageCount: number, turnsSinceUpdate: number): boolean {
  if (messageCount <= SESSION_SUMMARY_KEEP_RECENT) return false;
  return turnsSinceUpdate >= SESSION_SUMMARY_UPDATE_EVERY;
}
