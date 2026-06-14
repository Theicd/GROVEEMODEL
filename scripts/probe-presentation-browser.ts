/**
 * Presentation QA — runs through live UI at localhost:5173
 * Uses DOM + optional window.__groveeQa (dev bridge).
 *
 * IMPORTANT: Playwright opens its OWN browser window — not your existing tab.
 * If model is already loaded in YOUR tab, either:
 *   1) Run with QA_HEADLESS=0 to watch the Playwright window, or
 *   2) In YOUR tab console: await __groveeQa.ask("שאלה", { newChat: true, forceLlm: true })
 *
 * Env: QA_FORCE_LLM=1  QA_HEADLESS=0  QA_ASK_TIMEOUT_MS=360000
 */
import { writeFileSync, appendFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Page } from "playwright";
import { USER_PRESENTATION_QUERIES } from "../app/src/userPresentationQueries.ts";
import type { QaTurnResult } from "../app/src/qaChatBridge.ts";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dir, "..");
const LOG = join(ROOT, "PRESENTATION_QA_LOG.txt");
const REPORT = join(ROOT, "PRESENTATION_QA_REPORT.md");

const BASE = process.env.QA_URL ?? "http://127.0.0.1:5173/";
const FORCE_LLM = process.env.QA_FORCE_LLM !== "0";
const HEADLESS = process.env.QA_HEADLESS !== "0";
const ASK_TIMEOUT_MS = Number(process.env.QA_ASK_TIMEOUT_MS ?? "360000");
const MODEL_READY_MS = Number(process.env.QA_MODEL_READY_MS ?? "1200000");
const START = Math.max(1, Number(process.env.QA_START ?? "1"));
const LIMIT = Number(process.env.QA_LIMIT ?? "0") || 0;

type Row = {
  id: string;
  group: string;
  category: string;
  prompt: string;
  status: "pass" | "partial" | "fail";
  via: "bridge" | "dom";
} & QaTurnResult & { error?: string };

async function readPageStatus(page: Page): Promise<string> {
  return page.evaluate(() => {
    const statusEl = document.querySelector(".status-text, .load-status, .intro-text");
    const pct = document.querySelector(".load-pct, .progress-text");
    const input = document.querySelector<HTMLTextAreaElement>("#user-in");
    const bridge = typeof window.__groveeQa !== "undefined";
    const parts = [
      input?.disabled ? "input:disabled" : "input:enabled",
      bridge ? "bridge:yes" : "bridge:no",
      statusEl?.textContent?.trim().slice(0, 80) ?? "",
      pct?.textContent?.trim() ?? "",
    ];
    return parts.filter(Boolean).join(" · ");
  });
}

/** Wait until chat input is enabled (= model loaded in UI). */
async function waitForAppReady(page: Page): Promise<void> {
  const started = Date.now();
  let lastLog = 0;

  while (Date.now() - started < MODEL_READY_MS) {
    const ready = await page.evaluate(() => {
      const input = document.querySelector<HTMLTextAreaElement>("#user-in");
      return Boolean(input && !input.disabled);
    });
    if (ready) {
      const bridge = await page.evaluate(() => Boolean(window.__groveeQa?.ready?.()));
      console.log(`Ready (${Math.round((Date.now() - started) / 1000)}s) · bridge=${bridge}`);
      return;
    }

    if (Date.now() - lastLog > 15_000) {
      const elapsed = Math.round((Date.now() - started) / 1000);
      lastLog = Date.now();
      const status = await readPageStatus(page);
      const loadBtn = page.locator("button.load-btn");
      if (await loadBtn.isVisible().catch(() => false)) {
        console.log(`[${elapsed}s] Still loading — load button visible (${status})`);
      } else {
        console.log(`[${elapsed}s] Waiting for model… ${status}`);
      }
    }

    await page.waitForTimeout(2000);
  }

  throw new Error(`Model not ready after ${MODEL_READY_MS / 1000}s — check Playwright window / network`);
}

async function newChatDom(page: Page): Promise<void> {
  await page.evaluate(() => {
    const btn = document.querySelector<HTMLButtonElement>("button.new-chat");
    btn?.click();
  });
  await page.waitForTimeout(200);
}

async function askViaDom(page: Page, prompt: string): Promise<QaTurnResult & { error?: string }> {
  const started = Date.now();
  await newChatDom(page);

  const input = page.locator("#user-in");
  await input.fill(prompt);
  await page.locator("form").first().evaluate((f) => (f as HTMLFormElement).requestSubmit());

  await page.waitForSelector("button.in-stop", { timeout: 30_000 }).catch(() => {});
  await page.waitForFunction(
    () => !document.querySelector("button.in-stop"),
    undefined,
    { timeout: ASK_TIMEOUT_MS },
  );

  const reply = await page.evaluate(() => {
    const msgs = [...document.querySelectorAll(".msg")];
    for (let i = msgs.length - 1; i >= 0; i--) {
      const icon = msgs[i].querySelector(".msg-icon");
      if (icon?.textContent?.includes("AI") || icon?.classList.contains("ai")) {
        return msgs[i].querySelector(".msg-txt")?.textContent?.trim() ?? "";
      }
    }
    return "";
  });

  return {
    query: prompt,
    reply,
    replySource: "model",
    usedModel: reply.length > 0,
    webContextSent: "",
    modelPromptOut: "",
    modelResponseIn: reply,
    searchProviders: [],
    searchSummary: "",
    ms: Date.now() - started,
  };
}

async function askQuestion(page: Page, prompt: string, forceLlm: boolean): Promise<{ result: QaTurnResult & { error?: string }; via: "bridge" | "dom" }> {
  const hasBridge = await page.evaluate(() => typeof window.__groveeQa?.ask === "function");
  if (hasBridge) {
    try {
      const result = await page.evaluate(
        async ({ prompt, forceLlm, timeoutMs }) => {
          const timeout = new Promise<never>((_, rej) => {
            setTimeout(() => rej(new Error("timeout")), timeoutMs);
          });
          return Promise.race([
            window.__groveeQa!.ask(prompt, { forceLlm, newChat: true }),
            timeout,
          ]);
        },
        { prompt, forceLlm, timeoutMs: ASK_TIMEOUT_MS },
      );
      return { result, via: "bridge" };
    } catch {
      /* fall through to DOM */
    }
  }
  const result = await askViaDom(page, prompt);
  return { result, via: "dom" };
}

function grade(r: Row): Row["status"] {
  if (r.error || !r.reply?.trim()) return "fail";
  const hasSearch = (r.searchProviders?.length ?? 0) > 0 || r.webContextSent.trim().length > 80;
  const goodReply = r.reply.trim().length >= 40;
  if (goodReply && (r.usedModel || hasSearch || /Doom|משחק/i.test(r.prompt))) return "pass";
  if (goodReply) return "partial";
  return "fail";
}

const icon = (s: Row["status"]) => ({ pass: "✅", partial: "⚠️", fail: "❌" }[s]);

const queries = USER_PRESENTATION_QUERIES.slice(
  START - 1,
  LIMIT > 0 ? START - 1 + LIMIT : undefined,
);

writeFileSync(
  LOG,
  `PRESENTATION QA — ${new Date().toISOString()}\nURL: ${BASE}?forceLlm=${FORCE_LLM ? "1" : "0"}\nשאלות: ${queries.length}\n\n`,
  "utf8",
);

const url = `${BASE}${BASE.includes("?") ? "&" : "?"}${FORCE_LLM ? "forceLlm=1&" : ""}qa=chat`;
console.log(`Opening ${url}`);
console.log(`Playwright opens a SEPARATE browser — it must load Gemma (cached ~1-3 min, first time ~10+ min).`);
console.log(`Set QA_HEADLESS=0 to watch the window.\n`);
console.log(`${queries.length} questions\n`);

const browser = await chromium.launch({ headless: HEADLESS });
const page = await browser.newPage();
page.on("console", (msg) => {
  if (msg.type() === "error") console.error("[page]", msg.text().slice(0, 120));
});

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120_000 });

const loadBtn = page.locator("button.load-btn");
if (await loadBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
  console.log("Clicking «טען מודל מקומי»…");
  await loadBtn.click();
}

console.log("Waiting until chat input is enabled (model ready)…");
await waitForAppReady(page);
console.log("Starting questions…\n");

const rows: Row[] = [];

for (let i = 0; i < queries.length; i++) {
  const q = queries[i];
  const n = i + 1;
  process.stdout.write(`\n[${n}/${queries.length}] ${q.id} ${q.category} — ${q.prompt.slice(0, 50)}…\n`);

  let base: QaTurnResult & { error?: string };
  let via: "bridge" | "dom" = "dom";
  try {
    const out = await askQuestion(page, q.prompt, FORCE_LLM);
    base = out.result;
    via = out.via;
  } catch (err) {
    base = {
      query: q.prompt,
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

  const row: Row = {
    id: q.id,
    group: q.group,
    category: q.category,
    prompt: q.prompt,
    via,
    ...base,
    status: "fail",
  };
  row.status = grade(row);
  rows.push(row);

  const block = [
    `${"=".repeat(72)}`,
    `[${n}/${queries.length}] ${q.id} ${icon(row.status)} [${q.category}] via=${via}`,
    q.prompt,
    `מודל: ${row.usedModel ? "כן" : "לא"} · ${row.ms}ms`,
    row.searchProviders?.length ? `מקורות: ${row.searchProviders.join(", ")}` : "",
    row.error ? `שגיאה: ${row.error}` : "",
    "",
    "--- WEB CONTEXT ---",
    (row.webContextSent || "(ריק)").slice(0, 2000),
    "",
    "--- תשובת הממשק ---",
    (row.modelResponseIn || row.reply || "(ריק)").slice(0, 2000),
    "",
  ]
    .filter(Boolean)
    .join("\n");

  appendFileSync(LOG, block + "\n", "utf8");
  process.stdout.write(block + "\n");
}

await browser.close();

const counts = { pass: 0, partial: 0, fail: 0 };
for (const r of rows) counts[r.status]++;

let md = `# דוח בדיקת מצגת\n\n**תאריך:** ${new Date().toISOString()}\n\n`;
md += `| ✅ | ⚠️ | ❌ |\n|--:|--:|--:|\n| ${counts.pass} | ${counts.partial} | ${counts.fail} |\n\n`;
for (const r of rows) {
  md += `### ${r.id} ${icon(r.status)} ${r.prompt}\n\n`;
  md += `- via: ${r.via} · model: ${r.usedModel} · ${r.ms}ms\n\n`;
  md += "```\n" + (r.reply || "(ריק)").slice(0, 1200) + "\n```\n\n";
}
writeFileSync(REPORT, md, "utf8");

console.log(`\n✅ ${counts.pass}  ⚠️ ${counts.partial}  ❌ ${counts.fail}`);
console.log(`Log: ${LOG}\nReport: ${REPORT}`);
