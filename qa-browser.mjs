import { chromium } from "playwright";

async function run(url) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
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
      const loadButton = page.getByRole("button", { name: "Load model" });
      if (await loadButton.isVisible().catch(() => false)) {
        await loadButton.click();
        await page.waitForTimeout(6000);
        const status = await page.locator(".status").innerText().catch(() => "status not found");
        const progress = await page.locator("progress").getAttribute("value").catch(() => "n/a");
        result.details += `; status=${status}; progress=${progress}`;
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

const targets = ["http://127.0.0.1:4174/", "https://theicd.github.io/GROVEEMODEL/"];
const outputs = [];
for (const t of targets) outputs.push(await run(t));

console.log(JSON.stringify(outputs, null, 2));
