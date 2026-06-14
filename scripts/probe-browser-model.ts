/**
 * Browser E2E — submits each question via window.__groveeQa on localhost:5173/?qa=chat
 * Captures Gemma reply + WEB CONTEXT sent to the model.
 *
 * Env: QA_LIMIT=10  QA_START=1  QA_FORCE_LLM=1 (default)  QA_HEADLESS=1
 */
import { writeFileSync, appendFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { CAPABILITY_PROBE_QUERIES } from "../app/src/capabilityProbeQueries.ts";
import { LANDING_CAPABILITY_CHIPS } from "../app/src/chatLandingContent.ts";
import type { QaTurnResult } from "../app/src/qaChatBridge.ts";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dir, "..");

const LOG = join(ROOT, "MODEL_QA_LOG.txt");
const REPORT = join(ROOT, "MODEL_QA_REPORT.md");
const BASE = process.env.QA_URL ?? "http://127.0.0.1:5173/";
const LIMIT = Number(process.env.QA_LIMIT ?? "0") || 0;
const START = Math.max(1, Number(process.env.QA_START ?? "1"));
const FORCE_LLM = process.env.QA_FORCE_LLM !== "0";
const HEADLESS = process.env.QA_HEADLESS === "1";
const ASK_TIMEOUT_MS = Number(process.env.QA_ASK_TIMEOUT_MS ?? "240000");
const MODEL_READY_MS = Number(process.env.QA_MODEL_READY_MS ?? "900000");

function buildQueries() {
  const seen = new Set<string>();
  const out: Array<{ id: string; query: string; category: string }> = [];
  for (const p of CAPABILITY_PROBE_QUERIES) {
    const q = p.query.trim();
    if (seen.has(q)) continue;
    seen.add(q);
    out.push({ id: p.id, query: q, category: p.category });
  }
  let i = 0;
  for (const chip of LANDING_CAPABILITY_CHIPS) {
    const q = chip.prompt.trim();
    if (seen.has(q)) continue;
    seen.add(q);
    i++;
    out.push({ id: `L${String(i).padStart(2, "0")}`, query: q, category: chip.category });
  }
  return out;
}

const icon = (r: QaTurnResult & { error?: string }) => {
  if (r.error) return "❌";
  if (r.usedModel) return "🤖";
  if (r.replySource?.startsWith("canned")) return "📦";
  return "⚠️";
};

const all = buildQueries();
const slice = all.slice(START - 1, LIMIT > 0 ? START - 1 + LIMIT : undefined);

writeFileSync(
  LOG,
  `MODEL QA (browser + Gemma) — ${new Date().toISOString()}\nURL: ${BASE}?qa=chat\nforceLlm=${FORCE_LLM}\n${slice.length}/${all.length} questions\n\n`,
  "utf8",
);

const url = `${BASE}${BASE.includes("?") ? "&" : "?"}qa=chat${FORCE_LLM ? "&forceLlm=1" : ""}`;
console.log(`Opening ${url}`);
console.log(`Questions: ${slice.length} (from #${START})`);

const browser = await chromium.launch({ headless: HEADLESS, slowMo: HEADLESS ? 0 : 30 });
const page = await browser.newPage();

page.on("console", (msg) => {
  if (msg.type() === "error") console.error("[page]", msg.text());
});

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120_000 });

console.log("Waiting for Gemma model to load (may take several minutes on first run)…");
await page.waitForFunction(() => window.__groveeQa?.ready?.(), undefined, { timeout: MODEL_READY_MS });
console.log("Model ready.\n");

type Row = { id: string; category: string } & QaTurnResult & { error?: string };
const rows: Row[] = [];

for (let i = 0; i < slice.length; i++) {
  const { id, query, category } = slice[i];
  const n = i + 1;
  process.stdout.write(`\n[${n}/${slice.length}] ${id} — ${query.slice(0, 55)}…\n`);

  let result: Row;
  try {
    const evalResult = await page.evaluate(
      async ({ query, forceLlm, timeoutMs }) => {
        const timeout = new Promise<never>((_, rej) => {
          setTimeout(() => rej(new Error("ask timeout")), timeoutMs);
        });
        return Promise.race([
          window.__groveeQa!.ask(query, { forceLlm, newChat: true }),
          timeout,
        ]);
      },
      { query, forceLlm: FORCE_LLM, timeoutMs: ASK_TIMEOUT_MS },
    );
    result = { id, category, ...evalResult };
  } catch (err) {
    result = {
      id,
      category,
      query,
      reply: "",
      replySource: "unknown",
      usedModel: false,
      webContextSent: "",
      modelPromptOut: "",
      modelResponseIn: "",
      searchProviders: [],
      searchSummary: "",
      ms: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }

  rows.push(result);

  const block = [
    `${"=".repeat(72)}`,
    `[${n}/${slice.length}] ${id} ${icon(result)} ${query}`,
    `קטגוריה: ${category}`,
    `מקור: ${result.replySource} · מודל: ${result.usedModel ? "כן" : "לא"} · ${result.ms}ms`,
    result.searchProviders?.length ? `חיפוש: ${result.searchProviders.join(", ")}` : "חיפוש: —",
    result.error ? `שגיאה: ${result.error}` : "",
    "",
    "--- WEB CONTEXT (נשלח למודל) ---",
    (result.webContextSent || "(ריק)").slice(0, 2500),
    "",
    "--- תשובת Gemma / ממשק ---",
    (result.modelResponseIn || result.reply || "(ריק)").slice(0, 2500),
    "",
  ]
    .filter(Boolean)
    .join("\n");

  appendFileSync(LOG, block + "\n", "utf8");
  process.stdout.write(block + "\n");
}

await browser.close();

const modelOk = rows.filter((r) => r.usedModel && r.reply?.trim()).length;
const canned = rows.filter((r) => !r.usedModel && r.reply?.trim()).length;
const failed = rows.filter((r) => r.error || !r.reply?.trim()).length;

let md = `# בדיקת מודל בדפדפן (Gemma + WEB CONTEXT)\n\n`;
md += `**תאריך:** ${new Date().toISOString()}\n\n`;
md += `| מדד | כמות |\n|-----|------|\n`;
md += `| 🤖 Gemma ענה + יש תשובה | ${modelOk} |\n`;
md += `| 📦 תשובה קבועה (ללא מודל) | ${canned} |\n`;
md += `| ❌ נכשל / ריק | ${failed} |\n\n`;

md += `| # | ID | | שאלה | מודל? | מקור | מקורות חיפוש |\n`;
md += `|---|-----|-|------|-------|------|-------------|\n`;
rows.forEach((r, idx) => {
  md += `| ${idx + 1} | ${r.id} | ${icon(r)} | ${r.query.replace(/\|/g, "\\|").slice(0, 40)} | ${r.usedModel ? "כן" : "לא"} | ${r.replySource} | ${(r.searchProviders || []).join(", ").slice(0, 30)} |\n`;
});

md += `\n## פירוט\n\n`;
for (const r of rows) {
  md += `### ${r.id} — ${icon(r)} ${r.query.slice(0, 50)}\n\n`;
  md += `- **usedModel:** ${r.usedModel}\n`;
  md += `- **replySource:** ${r.replySource}\n`;
  md += `- **search:** ${r.searchProviders?.join(", ") || "—"}\n\n`;
  md += "**WEB CONTEXT:**\n\n```\n" + (r.webContextSent || "(ריק)").slice(0, 1500) + "\n```\n\n";
  md += "**תשובת מודל:**\n\n```\n" + (r.modelResponseIn || r.reply || "(ריק)").slice(0, 1500) + "\n```\n\n";
}

writeFileSync(REPORT, md, "utf8");

console.log(`\n${"=".repeat(72)}`);
console.log(`🤖 ${modelOk}  📦 ${canned}  ❌ ${failed}`);
console.log(`Log: ${LOG}`);
console.log(`Report: ${REPORT}`);
