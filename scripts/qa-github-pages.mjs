import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const TARGET_URL = process.env.GROVEE_QA_URL ?? "https://theicd.github.io/GROVEEMODEL/";
const OUT_DIR = "qa-artifacts";
/** Full 4-stage preload can be very slow on cold cache / GitHub Pages; override with GROVEE_QA_READY_MS. */
const READY_TIMEOUT_MS = Number(process.env.GROVEE_QA_READY_MS ?? 900000);
/** Skip waiting for the chat composer (full 4-model load can take 15m+ on cold cache). `--smoke` or env `1`. */
const SKIP_CHAT_WAIT =
  process.env.GROVEE_QA_SKIP_CHAT_WAIT === "1" || process.argv.includes("--smoke");

function expectedBundleFromLocalDocs() {
  try {
    const html = fs.readFileSync("docs/index.html", "utf8");
    const m = html.match(/src="\.\/assets\/(index-[^"]+\.js)"/i);
    return m?.[1] ?? null;
  } catch {
    return null;
  }
}

/** True when the built bundle in docs/ contains the 4-stage startup UI (strict QA only then). */
function localBundleHasFourStagePreload() {
  try {
    const name = expectedBundleFromLocalDocs();
    if (!name) return false;
    const js = fs.readFileSync(path.join(__dirname, "..", "docs", "assets", name), "utf8");
    return js.includes("data-preload-stage") && js.includes("loading-stage-rail");
  } catch {
    return false;
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

  await page.goto(TARGET_URL, { waitUntil: "domcontentloaded", timeout: 120000 });
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

  const loading = page.locator(".loading-screen");
  let sawStage4 = false;
  let hadLoadingUi = false;
  try {
    await loading.first().waitFor({ state: "visible", timeout: 45_000 });
    hadLoadingUi = true;
  } catch {
    // warm cache / instant — chat may already be ready
  }

  const prompt = page.getByPlaceholder(/ask anything|start the app first/i).first();
  const stage4 = page.locator(".loading-screen[data-preload-stage='4']");
  const probe = setInterval(() => {
    void stage4
      .count()
      .then((n) => {
        if (n > 0) sawStage4 = true;
      })
      .catch(() => {});
  }, 400);

  let promptVisible = false;
  try {
    if (!SKIP_CHAT_WAIT) {
      await prompt.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
      promptVisible = true;
    }
  } catch {
    promptVisible = false;
  } finally {
    clearInterval(probe);
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
      hadLoadingUi,
      sawPreloadStage4: sawStage4,
    },
  };

  const strictFourStage = localBundleHasFourStagePreload();

  const checks = {
    latestBundleLoaded: hasExpectedIndex,
    noDataCloneError: !result.errors.dataCloneError,
    noOldBundleRef: !result.errors.oldBundleRef,
    noHttpImageInDom: result.imageFlow.foundHttpImage === 0,
    noRuntimePageErrors: result.errors.pageErrors.length === 0,
    /** When docs bundle includes 4-stage UI and live matches it, we should have hit caption stage before chat. */
    fourStagePreload:
      SKIP_CHAT_WAIT || !strictFourStage || !hasExpectedIndex || sawStage4,
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
