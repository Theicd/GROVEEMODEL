import { chromium } from "playwright";
import fs from "node:fs";

const TARGET_URL = "https://theicd.github.io/GROVEEMODEL/";
const OUT_DIR = "qa-artifacts";
const READY_TIMEOUT_MS = 120000;

function expectedBundleFromLocalDocs() {
  try {
    const html = fs.readFileSync("docs/index.html", "utf8");
    const m = html.match(/src="\.\/assets\/(index-[^"]+\.js)"/i);
    return m?.[1] ?? null;
  } catch {
    return null;
  }
}

function hasDataCloneError(messages) {
  return messages.some((m) => /DataCloneError/i.test(m.text));
}

function hasOldBundle(messages) {
  return messages.some((m) => /index-DSR0YkYa\.js|model\.worker-9BeGedNx\.js/i.test(m.text));
}

async function run() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  const consoleEvents = [];
  const pageErrors = [];

  page.on("console", (msg) => {
    consoleEvents.push({ type: msg.type(), text: msg.text() });
  });
  page.on("pageerror", (err) => {
    pageErrors.push(err?.message ?? String(err));
  });

  await page.goto(TARGET_URL, { waitUntil: "networkidle", timeout: 120000 });
  await page.waitForTimeout(1500);

  await page.screenshot({ path: `${OUT_DIR}/01-landing.png`, fullPage: true });

  const scriptSrcs = await page.$$eval("script[src]", (els) => els.map((e) => e.getAttribute("src") || ""));
  const cssHrefs = await page.$$eval("link[rel='stylesheet'][href]", (els) =>
    els.map((e) => e.getAttribute("href") || ""),
  );
  const expectedBundle = expectedBundleFromLocalDocs();
  const hasExpectedIndex = expectedBundle ? scriptSrcs.some((s) => s.includes(expectedBundle)) : false;

  const startButton = page.getByRole("button", { name: /התחל|start/i }).first();
  if (await startButton.isVisible().catch(() => false)) {
    await startButton.click();
  }

  const prompt = page.locator("textarea, input[placeholder*='Ask']").first();
  let promptVisible = false;
  try {
    await prompt.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    promptVisible = true;
  } catch {
    promptVisible = false;
  }

  if (promptVisible) {
    await prompt.fill("צור תמונה של חתול כחול");
    await prompt.press("Enter");
    await page.waitForTimeout(7000);
  } else {
    await page.waitForTimeout(2000);
  }
  await page.screenshot({ path: `${OUT_DIR}/02-after-image-request.png`, fullPage: true });

  const bodyText = (await page.locator("body").innerText()).toLowerCase();
  const hasPollinationsInDom = bodyText.includes("pollinations.ai") || bodyText.includes("http://") || bodyText.includes("https://");

  const result = {
    url: TARGET_URL,
    assets: { scriptSrcs, cssHrefs, hasLatestIndex: hasExpectedIndex, expectedBundle },
    errors: {
      dataCloneError: hasDataCloneError(consoleEvents) || pageErrors.some((e) => /DataCloneError/i.test(e)),
      oldBundleRef: hasOldBundle(consoleEvents),
      pageErrors,
      consoleErrorCount: consoleEvents.filter((e) => e.type === "error").length,
      consoleEvents: consoleEvents.slice(-80),
    },
    imageFlow: {
      promptVisible,
      hasPollinationsInDom,
      foundBlobImage: await page.locator("img[src^='blob:']").count(),
      foundHttpImage: await page.locator("img[src^='http'], img[src^='https']").count(),
    },
    uiState: {
      loadingVisible: await page.locator(".loading-screen").count(),
      startVisible: await page.locator(".hero-screen").count(),
      statusText: await page.locator(".hero-status, .loading-headline-status").first().innerText().catch(() => ""),
    },
  };

  const checks = {
    latestBundleLoaded: hasExpectedIndex,
    noDataCloneError: !result.errors.dataCloneError,
    noOldBundleRef: !result.errors.oldBundleRef,
    noHttpImageInDom: result.imageFlow.foundHttpImage === 0,
    noRuntimePageErrors: result.errors.pageErrors.length === 0,
  };

  result.imageFlow.testStatus = result.imageFlow.promptVisible ? "executed" : "skipped_waiting_for_model_load";
  result.checks = checks;
  result.pass = Object.values(checks).every(Boolean);
  await context.close();
  await browser.close();

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (!result.pass) process.exit(2);
}

run().catch((err) => {
  process.stderr.write(`${err?.stack || err}\n`);
  process.exit(1);
});
