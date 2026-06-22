import type { QaTurnResult } from "./qaChatBridge";
import type { UserPresentationQuery } from "./userPresentationQueries";
import { autoGradePresentationQuery } from "./presentationQaExpectations";

export type QaManualStatus = "pass" | "partial" | "fail" | "skip" | "untested";

export type QaRecordedResult = QaTurnResult & {
  autoStatus: Exclude<QaManualStatus, "untested" | "skip">;
  manualStatus?: Exclude<QaManualStatus, "untested">;
  testedAt: number;
  runVersion?: string;
  note?: string;
};

/** Bump when QA pipeline / canned logic changes — stale results get ⚠️ in UI. */
export const QA_CODE_VERSION = "2026-06-15-ships-crisp-v17";

export const QA_STATUS_ICON: Record<QaManualStatus, string> = {
  pass: "✅",
  partial: "⚠️",
  fail: "❌",
  skip: "⏭️",
  untested: "○",
};

export function autoGradeResult(
  r: QaTurnResult,
  q?: UserPresentationQuery,
): Exclude<QaManualStatus, "untested" | "skip"> {
  if (q) return autoGradePresentationQuery(q, r);
  if (r.error || !r.reply?.trim()) return "fail";
  const hasSearch =
    (r.searchProviders?.length ?? 0) > 0 || r.webContextSent.trim().length > 80;
  const goodReply = r.reply.trim().length >= 40;
  if (goodReply && (r.usedModel || hasSearch || /Doom|משחק/i.test(r.query))) return "pass";
  if (goodReply) return "partial";
  return "fail";
}

export { autoGradePresentationQuery } from "./presentationQaExpectations";

export function effectiveStatus(r: QaRecordedResult | undefined): QaManualStatus {
  if (!r) return "untested";
  return r.manualStatus ?? r.autoStatus;
}

export function buildPresentationQaReport(
  queries: UserPresentationQuery[],
  results: Record<string, QaRecordedResult>,
): string {
  const counts = { pass: 0, partial: 0, fail: 0, skip: 0, untested: 0 };
  for (const q of queries) {
    counts[effectiveStatus(results[q.id])]++;
  }

  const lines: string[] = [
    `# דוח בדיקת מצגת GROVEE`,
    ``,
    `**תאריך:** ${new Date().toISOString()}`,
    ``,
    `| ✅ | ⚠️ | ❌ | ⏭️ | ○ |`,
    `|--:|--:|--:|--:|--:|`,
    `| ${counts.pass} | ${counts.partial} | ${counts.fail} | ${counts.skip} | ${counts.untested} |`,
    ``,
  ];

  for (const q of queries) {
    const r = results[q.id];
    const status = effectiveStatus(r);
    lines.push(`## ${q.id} ${QA_STATUS_ICON[status]} [${q.category}]`);
    lines.push(q.prompt);
    lines.push(``);
    if (!r) {
      lines.push(`_(לא נבדק)_`);
      lines.push(``);
      continue;
    }
    lines.push(
      `- auto: ${r.autoStatus} · effective: ${status} · source: ${r.replySource} · model: ${r.usedModel ? "כן" : "לא"} · ${r.ms}ms`,
    );
    lines.push(`- testedAt: ${new Date(r.testedAt).toISOString()} · runVersion: ${r.runVersion ?? "(none)"}`);
    if (r.runVersion && r.runVersion !== QA_CODE_VERSION) {
      lines.push(`- ⚠️ stale — recorded under older code (${r.runVersion})`);
    }
    if (r.searchProviders?.length) {
      lines.push(`- מקורות: ${r.searchProviders.join(", ")}`);
    }
    if (r.note?.trim()) lines.push(`- הערה: ${r.note.trim()}`);
    if (r.error) lines.push(`- שגיאה: ${r.error}`);
    lines.push(``);
    lines.push(`**תשובה:**`);
    lines.push((r.reply || "(ריק)").slice(0, 1500));
    if (r.webContextSent.trim()) {
      lines.push(``);
      lines.push(`**WEB CONTEXT (קטע):**`);
      lines.push(r.webContextSent.slice(0, 800));
    }
    lines.push(``);
    lines.push(`---`);
    lines.push(``);
  }

  return lines.join("\n");
}

export const QA_RESULTS_STORAGE_KEY = "grovee-presentation-qa-results-v3";

export function clearQaResults(): void {
  try {
    localStorage.removeItem(QA_RESULTS_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export function loadQaResults(): Record<string, QaRecordedResult> {
  try {
    const raw = localStorage.getItem(QA_RESULTS_STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as Record<string, QaRecordedResult>;
  } catch {
    return {};
  }
}

export function saveQaResults(results: Record<string, QaRecordedResult>): void {
  localStorage.setItem(QA_RESULTS_STORAGE_KEY, JSON.stringify(results));
}
