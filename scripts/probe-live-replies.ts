/**
 * Live end-to-end QA — simulates App.tsx reply path per question.
 * Runs sequentially, prints + appends each result (question → structured answer).
 */
import { appendFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { CAPABILITY_PROBE_QUERIES } from "../app/src/capabilityProbeQueries.ts";
import { LANDING_CAPABILITY_CHIPS } from "../app/src/chatLandingContent.ts";
import { runWebSearch } from "../app/src/webSearch/orchestrator.ts";
import { classifySearchIntents, needsWebSearch } from "../app/src/webSearch/intents.ts";
import { buildCapabilityLiveReply } from "../app/src/webSearch/capabilityReplyMessages.ts";
import { buildGlobeCommand } from "../app/src/realityGlobe/intents.ts";
import { buildGlobePlaceReply } from "../app/src/realityGlobe/globePresentation.ts";
import {
  isGameSearchRequest,
  parseGameUserRequest,
} from "../app/src/gameSearch/gameIntents.ts";
import {
  buildGameSearchFoundReply,
  buildGameSearchNotFoundReply,
} from "../app/src/gameSearch/gameReplyMessages.ts";
import { searchOnlineGamesWithFallback } from "../app/src/gameSearch/archiveBrowser.ts";

const __dir = dirname(fileURLToPath(import.meta.url));
const LOG_PATH = join(__dir, "..", "LIVE_QA_LOG.txt");
const REPORT_PATH = join(__dir, "..", "LIVE_QA_REPORT.md");

type Status = "pass" | "partial" | "fail" | "skip";

type LiveRow = {
  idx: number;
  id: string;
  query: string;
  status: Status;
  path: string;
  intents: string;
  providers: string;
  ms: number;
  reply: string;
  error?: string;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const icon = (s: Status) => ({ pass: "✅", partial: "⚠️", fail: "❌", skip: "⏭️" }[s]);

function buildQueryList(): Array<{ id: string; query: string }> {
  const seen = new Set<string>();
  const out: Array<{ id: string; query: string }> = [];
  for (const p of CAPABILITY_PROBE_QUERIES) {
    const q = p.query.trim();
    if (seen.has(q)) continue;
    seen.add(q);
    out.push({ id: p.id, query: q });
  }
  let chipIdx = 0;
  for (const chip of LANDING_CAPABILITY_CHIPS) {
    const q = chip.prompt.trim();
    if (seen.has(q)) continue;
    seen.add(q);
    chipIdx++;
    out.push({ id: `L${String(chipIdx).padStart(2, "0")}`, query: q });
  }
  return out;
}

async function simulateUiReply(query: string): Promise<Omit<LiveRow, "idx" | "id" | "query">> {
  const started = performance.now();
  const intents = classifySearchIntents(query);
  const intentStr = intents.join(", ") || "—";

  if (isGameSearchRequest(query)) {
    const req = parseGameUserRequest(query);
    try {
      const result = await searchOnlineGamesWithFallback(req, 8);
      const reply =
        result.matchFound && result.games.length
          ? buildGameSearchFoundReply(result.games.length, req)
          : buildGameSearchNotFoundReply(req);
      return {
        status: result.matchFound && result.games.length ? "pass" : "partial",
        path: "game-panel + archive.org",
        intents: intentStr,
        providers: result.matchFound ? "archive.org" : "—",
        ms: Math.round(performance.now() - started),
        reply,
      };
    } catch (err) {
      return {
        status: "fail",
        path: "game-panel",
        intents: intentStr,
        providers: "—",
        ms: Math.round(performance.now() - started),
        reply: "",
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  const globeCmd = buildGlobeCommand(query, intents);
  if (globeCmd?.type === "focusPlaceQuiet" && globeCmd.presentation !== false) {
    const reply = buildGlobePlaceReply(globeCmd.name);
    return {
      status: "pass",
      path: "globe-place (canned)",
      intents: intentStr,
      providers: "—",
      ms: Math.round(performance.now() - started),
      reply,
    };
  }

  const shouldSearch = needsWebSearch(query);
  if (shouldSearch) {
    try {
      const search = await runWebSearch(query);
      const okSources = search.sources.filter((s) => s.ok && s.text.trim());
      const providers = okSources.map((s) => s.provider).join(", ") || "—";
      const canned = buildCapabilityLiveReply(query, search.intents, search.sources);

      if (canned) {
        return {
          status: "pass",
          path: "canned-live-reply",
          intents: search.intents.join(", ") || intentStr,
          providers,
          ms: Math.round(performance.now() - started),
          reply: canned,
        };
      }

      if (globeCmd) {
        const layerReply = `פתחתי את פאנל עולם חי (REALITY LIVE) — פקודה: ${globeCmd.type}${
          "layer" in globeCmd ? ` · שכבה: ${globeCmd.layer}` : ""
        }.`;
        return {
          status: okSources.length ? "pass" : "partial",
          path: `globe-${globeCmd.type}`,
          intents: search.intents.join(", ") || intentStr,
          providers,
          ms: Math.round(performance.now() - started),
          reply: layerReply,
        };
      }

      if (okSources.length) {
        const fallback = okSources
          .slice(0, 2)
          .map((s) => `[${s.label}]\n${s.text.trim().slice(0, 400)}`)
          .join("\n\n");
        return {
          status: "partial",
          path: "search-brief-only (would use LLM)",
          intents: search.intents.join(", ") || intentStr,
          providers,
          ms: Math.round(performance.now() - started),
          reply: fallback,
        };
      }

      const err = search.sources.find((s) => !s.ok)?.error ?? "אין מקורות";
      return {
        status: "fail",
        path: "web-search",
        intents: search.intents.join(", ") || intentStr,
        providers: "—",
        ms: Math.round(performance.now() - started),
        reply: "",
        error: err,
      };
    } catch (err) {
      return {
        status: "fail",
        path: "web-search",
        intents: intentStr,
        providers: "—",
        ms: Math.round(performance.now() - started),
        reply: "",
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  if (globeCmd) {
    return {
      status: "pass",
      path: `globe-${globeCmd.type}`,
      intents: intentStr,
      providers: "—",
      ms: Math.round(performance.now() - started),
      reply: `פאנל גלובוס: ${globeCmd.type}`,
    };
  }

  return {
    status: "skip",
    path: "no-search",
    intents: intentStr,
    providers: "—",
    ms: Math.round(performance.now() - started),
    reply: "לא מזוהה כשאילתת נתונים חיים — תלוי LLM בלבד.",
  };
}

function formatBlock(row: LiveRow): string {
  const lines = [
    `${"=".repeat(72)}`,
    `[${row.idx}/${row.id}] ${icon(row.status)} ${row.query}`,
    `נתיב: ${row.path} · ${row.ms}ms`,
    `Intents: ${row.intents}`,
    `מקורות: ${row.providers}`,
  ];
  if (row.error) lines.push(`שגיאה: ${row.error}`);
  lines.push("", "--- תשובת הממשק ---", row.reply.trim() || "(ריק)", "");
  return lines.join("\n");
}

const queries = buildQueryList();
writeFileSync(
  LOG_PATH,
  `LIVE QA — ${new Date().toISOString()}\nסה"כ ${queries.length} שאלות\n\n`,
  "utf8",
);

const rows: LiveRow[] = [];
let idx = 0;

for (const { id, query } of queries) {
  idx++;
  process.stdout.write(`\n[${idx}/${queries.length}] ${id} — ${query.slice(0, 50)}…\n`);
  const result = await simulateUiReply(query);
  const row: LiveRow = { idx, id, query, ...result };
  rows.push(row);
  const block = formatBlock(row);
  appendFileSync(LOG_PATH, block, "utf8");
  process.stdout.write(block);
  await sleep(150);
}

const counts = { pass: 0, partial: 0, fail: 0, skip: 0 };
for (const r of rows) counts[r.status]++;

let md = `# בדיקה חיה — תשובות ממשק\n\n`;
md += `**תאריך:** ${new Date().toISOString()}\n\n`;
md += `| סטטוס | כמות |\n|--------|------|\n`;
md += `| ✅ תשובה מובנית מלאה | ${counts.pass} |\n`;
md += `| ⚠️ חלקי / LLM | ${counts.partial} |\n`;
md += `| ❌ נכשל | ${counts.fail} |\n`;
md += `| ⏭️ דילוג | ${counts.skip} |\n\n`;
md += `**סה"כ:** ${rows.length} שאלות\n\n`;
md += `## רשימה מהירה\n\n`;
md += `| # | ID | סטטוס | שאלה | נתיב | מקורות |\n`;
md += `|---|-----|--------|------|------|--------|\n`;
for (const r of rows) {
  const q = r.query.replace(/\|/g, "\\|").slice(0, 45);
  md += `| ${r.idx} | ${r.id} | ${icon(r.status)} | ${q} | ${r.path} | ${r.providers.slice(0, 40)} |\n`;
}
md += `\n## פירוט מלא\n\n`;
for (const r of rows) {
  md += `### ${r.idx}. ${r.id} — ${icon(r.status)} ${r.query.slice(0, 60)}\n\n`;
  md += `- **נתיב:** ${r.path}\n`;
  md += `- **Intents:** ${r.intents}\n`;
  md += `- **מקורות:** ${r.providers}\n`;
  md += `- **זמן:** ${r.ms}ms\n\n`;
  if (r.error) md += `**שגיאה:** ${r.error}\n\n`;
  md += "```\n" + (r.reply.trim() || "(ריק)") + "\n```\n\n";
}

writeFileSync(REPORT_PATH, md, "utf8");

console.log(`\n${"=".repeat(72)}`);
console.log(`סיכום: ✅ ${counts.pass}  ⚠️ ${counts.partial}  ❌ ${counts.fail}  ⏭️ ${counts.skip}`);
console.log(`Log: ${LOG_PATH}`);
console.log(`Report: ${REPORT_PATH}`);
process.exit(counts.fail > 20 ? 1 : 0);
