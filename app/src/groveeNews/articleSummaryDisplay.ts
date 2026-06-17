// @ts-nocheck
/** App-owned display helpers — kept outside engine/ so sync-grovee-news does not wipe them. */
import type { SummarizerResult } from "./engine/summarize/summarizerClient";
import { cleanSummaryText } from "./engine/summarize/summaryQuality";

const LABEL_JUNK =
  /(?:^|\s)(?:EN|כותרת\s*EN|title_en|summary|facts?|entities?|keywords?|עובדה\s*\d+|fact\s*\d+)\s*:?\s*/gi;

/** Remove structural labels Qwen / translate sometimes leak into user text. */
export function stripSummaryMarkers(text: string): string {
  return cleanSummaryText(
    text
      .replace(/\r\n/g, "\n")
      .replace(LABEL_JUNK, " ")
      .replace(/\bעובדה\s+\d+\s*:/gi, " ")
      .replace(/\bFACT\s+\d+\s*:/gi, " ")
      .replace(/\bEN\s*:/gi, " ")
      .replace(/\s{2,}/g, " ")
      .replace(/\s+([,.])/g, "$1")
      .trim(),
  );
}

export function buildBriefNotes(result: SummarizerResult, fallbackTitle: string): string {
  const lines: string[] = [];
  const title = stripSummaryMarkers(result.titleEn || fallbackTitle);
  const summary = stripSummaryMarkers(result.summary);
  if (title) lines.push(`Title: ${title}`);
  if (summary) lines.push(`Summary: ${summary}`);
  for (const fact of result.keyFacts.slice(0, 4)) {
    const f = stripSummaryMarkers(fact);
    if (f.length > 10) lines.push(`- ${f}`);
  }
  return lines.join("\n");
}

type BriefLabels = { title: string; summary: string };

const LABELS: Record<string, BriefLabels> = {
  he: { title: "כותרת", summary: "תקציר" },
  en: { title: "Title", summary: "Summary" },
};

function labelsFor(lang: string): BriefLabels {
  return LABELS[lang] ?? LABELS.en;
}

/** Deterministic fallback when Gemma polish is weak. */
export function formatArticleSummaryForUser(
  result: SummarizerResult,
  fallbackTitle: string,
  targetLang: string,
): string {
  const labels = labelsFor(targetLang);
  const title = stripSummaryMarkers(result.titleEn || fallbackTitle);
  const summary = stripSummaryMarkers(result.summary);
  const facts = result.keyFacts.map(stripSummaryMarkers).filter((f) => f.length > 12);

  let body = summary;
  if (facts.length && (!body || body.length < 40)) {
    body = facts.slice(0, 3).join(". ") + ".";
  } else if (facts.length && body.length < 120) {
    const extra = facts.find((f) => !body.includes(f.slice(0, 24)));
    if (extra) body = `${body} ${extra}`.trim();
  }

  if (!title && !body) return "";

  const parts: string[] = [];
  if (title) parts.push(`${labels.title}: ${title}`);
  if (body) {
    if (parts.length) parts.push("");
    parts.push(`${labels.summary}: ${body}`);
  }
  return parts.join("\n").trim();
}

/** Clean model rephrase output — keep כותרת/תקציר structure, drop junk. */
export function cleanRephrasedBrief(raw: string, targetLang: string): string {
  const text = raw.replace(/\r\n/g, "\n").trim();
  if (!text) return "";

  const labels = labelsFor(targetLang);
  const titleRe = new RegExp(`^(?:${labels.title}|כותרת|Title)\\s*:\\s*(.+)$`, "im");
  const summaryRe = new RegExp(`^(?:${labels.summary}|תקציר|Summary)\\s*:\\s*([\\s\\S]+)$`, "im");

  const titleMatch = text.match(titleRe);
  const summaryMatch = text.match(summaryRe);

  if (titleMatch || summaryMatch) {
    const parts: string[] = [];
    if (titleMatch?.[1]) {
      parts.push(`${labels.title}: ${stripSummaryMarkers(titleMatch[1])}`);
    }
    if (summaryMatch?.[1]) {
      if (parts.length) parts.push("");
      const body = stripSummaryMarkers(summaryMatch[1].replace(/\n+/g, " "));
      if (body) parts.push(`${labels.summary}: ${body}`);
    }
    const out = parts.join("\n").trim();
    if (out.length > 30) return out;
  }

  const stripped = stripSummaryMarkers(text.replace(/\n+/g, " "));
  if (stripped.length < 25) return "";
  return formatArticleSummaryForUser(
    { summary: stripped, keyFacts: [], entities: [], keywords: [] },
    "",
    targetLang,
  );
}
