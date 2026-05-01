import { chromium } from "playwright";

async function run(url) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  const hfRequests = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("requestfinished", (req) => {
    const target = req.url();
    if (target.includes("huggingface.co") || target.includes("cdn-lfs")) {
      hfRequests.push(target);
    }
  });

  const result = { url, ok: false, details: "", errors: [] };
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(1000);
    const title = await page.title();

    const hasAppHeader = await page.locator("h1").first().isVisible().catch(() => false);
    result.ok = hasAppHeader;
    result.details = `title=${title}; headerVisible=${hasAppHeader}`;

    if (hasAppHeader) {
      const loadButton = page
        .getByRole("button", { name: /^(Start|Load model|Load Gemma 4)$/ })
        .first();
      if (await loadButton.isVisible().catch(() => false)) {
        await loadButton.click();
        await page.waitForSelector(".loading-screen .status-line", { timeout: 20000 }).catch(() => {});
        await page.waitForTimeout(10000);
        const status = await page.locator(".loading-screen .status-line").first().innerText().catch(() => "status not found");
        const progress = await page.locator(".loading-screen .percent").innerText().catch(() => "n/a");
        const detail = await page
          .locator(".loading-screen .status-line.secondary")
          .first()
          .innerText()
          .catch(() => "detail not found");
        const file = await page
          .locator(".loading-screen .status-line.secondary")
          .nth(1)
          .innerText()
          .catch(() => "file not found");
        result.details += `; status=${status}; progress=${progress}; detail=${detail}; file=${file}; hfRequests=${hfRequests.length}`;
        const readyShell = page.locator(".ready-shell");
        await readyShell.waitFor({ timeout: 360000 }).catch(() => {});
        if (await readyShell.isVisible().catch(() => false)) {
          const promptBox = page.locator("textarea").first();
          await promptBox.fill("Hi, are you ready?");
          await page.locator(".send-btn").click();
          await page.waitForTimeout(12000);
          const bubbles = await page.locator(".bubble").count();
          const assistantLast = await page
            .locator(".bubble.assistant p")
            .last()
            .innerText()
            .catch(() => "");
          result.details += `; bubbles=${bubbles}; assistantLastLen=${assistantLast.trim().length}`;
          result.ok = hfRequests.length > 0 && bubbles >= 2 && assistantLast.trim().length > 0;
        } else {
          result.ok = hfRequests.length > 0;
        }
      }
    }
  } catch (e) {
    result.ok = false;
    result.details = e instanceof Error ? e.message : String(e);
  } finally {
    result.errors = errors.slice(0, 5);
    await browser.close();
  }
  return result;
}

const targets = ["http://127.0.0.1:4173/GROVEEMODEL/", "https://theicd.github.io/GROVEEMODEL/?v=e2e"];
const outputs = [];
for (const t of targets) outputs.push(await run(t));

console.log(JSON.stringify(outputs, null, 2));
