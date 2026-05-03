/**
 * GROVEE — סדרת בדיקות QA (Playwright)
 *
 * בודק:
 * - זמן עד שהממשק מופיע (DOM)
 * - כפתור «נקה מטמון»: סטטוס הצלחה + ירידה במפתחות Cache Storage רלוונטיים + Storage estimate (ככל שדפדפן מדווח)
 * - מצב `--full`: זמן עד צ'אט מוכן, זמן עד תשובה ל־«היי», זמן עד תמונה מקומית + אין img עם http(s) בבועות
 *
 * משתני סביבה:
 *   GROVEE_QA_URL          (ברירת מחדל: http://127.0.0.1:4173/)
 *   GROVEE_QA_READY_MS     timeout לטעינת מודלים (ברירת מחדל: 900000)
 *   GROVEE_QA_HI_MS        timeout לתשובת היי (ברירת מחדל: 120000)
 *   GROVEE_QA_IMAGE_MS     timeout ליצירת תמונה (ברירת מחדל: 300000)
 *
 * שימוש:
 *   npm run qa:e2e              # ניקוי מטמון + מדדים מהירים (ללא הורדת מודלים)
 *   npm run qa:e2e:full         # כולל התחל + היי + תמונה (ארוך מאוד)
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, "..", "qa-artifacts");

const BASE_URL = process.env.GROVEE_QA_URL ?? "http://127.0.0.1:4173/";
const READY_MS = Number(process.env.GROVEE_QA_READY_MS ?? 900_000);
const HI_MS = Number(process.env.GROVEE_QA_HI_MS ?? 120_000);
const IMAGE_MS = Number(process.env.GROVEE_QA_IMAGE_MS ?? 300_000);
const FULL = process.argv.includes("--full") || process.env.GROVEE_QA_E2E_FULL === "1";

async function storageSnapshot(page) {
  return page.evaluate(async () => {
    let cacheKeys = [];
    try {
      if ("caches" in globalThis) cacheKeys = await caches.keys();
    } catch {
      cacheKeys = [];
    }
    const modelishKeys = cacheKeys.filter((k) => {
      const l = k.toLowerCase();
      return (
        l.includes("transformers") ||
        l.includes("huggingface") ||
        l.startsWith("hf-") ||
        l.includes("onnx") ||
        l.includes("ort-wasm") ||
        l.includes("ort.") ||
        l.includes("web-txt2img") ||
        l.includes("txt2img")
      );
    });
    let usage = null;
    let quota = null;
    try {
      if (navigator.storage?.estimate) {
        const e = await navigator.storage.estimate();
        usage = typeof e.usage === "number" ? e.usage : null;
        quota = typeof e.quota === "number" ? e.quota : null;
      }
    } catch {
      /* ignore */
    }
    return { cacheKeys, modelishCacheCount: modelishKeys.length, modelishCacheKeys: modelishKeys.slice(0, 40), usage, quota };
  });
}

async function countChatHttpImages(page) {
  return page.$$eval(".bubble img, .msg-body img", (imgs) =>
    imgs.filter((img) => {
      const s = (img.getAttribute("src") || "").trim().toLowerCase();
      return s.startsWith("http://") || s.startsWith("https://");
    }).length,
  );
}

async function countChatBlobImages(page) {
  return page.$$eval(".bubble img, .msg-body img", (imgs) =>
    imgs.filter((img) => {
      const s = (img.getAttribute("src") || "").trim();
      return s.startsWith("blob:");
    }).length,
  );
}

async function run() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const tRun0 = Date.now();
  const report = {
    url: BASE_URL,
    mode: FULL ? "full" : "smoke",
    timingsMs: {},
    storage: {},
    tests: [],
    pass: true,
  };

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));

  const pushTest = (name, ok, detail = "") => {
    report.tests.push({ name, ok, detail });
    if (!ok) report.pass = false;
  };

  try {
    const tNav = Date.now();
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 120_000 });
    await page.waitForTimeout(800);
    report.timingsMs.domContentLoaded = Date.now() - tNav;

    await page.waitForSelector("h1.hero-brand, h1, .hero-brand", { timeout: 30_000 });
    report.timingsMs.uiShellVisible = Date.now() - tNav;
    pushTest("ui_shell_visible", true, "GROVEE / hero");

    const snap0 = await storageSnapshot(page);
    report.storage.beforeClear = snap0;

    /** ניקוי מטמון מהמסך הראשי (בלי טעינת מודלים) */
    const clearBtn = page.getByTestId("grovee-clear-cache").first();
    if (await clearBtn.isVisible().catch(() => false)) {
      await clearBtn.click();
      await page.getByText(/המטמון נוקה|ניקוי מטמון נכשל/).waitFor({ timeout: 120_000 });
      const status = await page.locator(".hero-status, .top-status").first().innerText().catch(() => "");
      const clearedOk = /המטמון נוקה/.test(status);
      pushTest("clear_cache_status_success", clearedOk, status.slice(0, 200));
    } else {
      pushTest("clear_cache_button_visible", false, "grovee-clear-cache not found");
    }

    const snap1 = await storageSnapshot(page);
    report.storage.afterClear = snap1;
    const keysDropped = snap0.modelishCacheCount >= snap1.modelishCacheCount;
    pushTest(
      "modelish_cache_keys_not_increased_after_clear",
      keysDropped,
      `before=${snap0.modelishCacheCount} after=${snap1.modelishCacheCount}`,
    );
    if (snap0.usage != null && snap1.usage != null) {
      const dropped = snap1.usage <= snap0.usage * 1.05;
      pushTest(
        "storage_estimate_usage_stable_or_lower",
        dropped,
        `usage before=${snap0.usage} after=${snap1.usage} (דפדפן; לא תמיד משקף דיסק פיזי)`,
      );
    } else {
      pushTest("storage_estimate_usage_stable_or_lower", true, "estimate() unavailable");
    }

    if (!FULL) {
      await page.screenshot({ path: path.join(OUT_DIR, "e2e-smoke.png"), fullPage: true });
      report.timingsMs.total = Date.now() - tRun0;
      await browser.close();
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
      process.exit(report.pass ? 0 : 2);
      return;
    }

    /** --- מצב מלא: טעינת מודלים + היי + תמונה --- */
    const tLoad0 = Date.now();
    await page.getByTestId("grovee-start").click();
    await page.getByTestId("grovee-prompt").waitFor({ state: "visible", timeout: READY_MS });
    report.timingsMs.toChatComposerVisible = Date.now() - tLoad0;

    const snapLoaded = await storageSnapshot(page);
    report.storage.afterModelsLoaded = snapLoaded;

    await page.getByTestId("grovee-prompt").fill("היי");
    const tHi0 = Date.now();
    await page.getByTestId("grovee-send").click();
    await page.waitForFunction(
      () => {
        const bubbles = document.querySelectorAll(".bubble.assistant");
        if (bubbles.length === 0) return false;
        const last = bubbles[bubbles.length - 1];
        return (last.textContent || "").trim().length > 2;
      },
      null,
      { timeout: HI_MS },
    );
    report.timingsMs.hiToAssistantReply = Date.now() - tHi0;

    const httpBeforeImg = await countChatHttpImages(page);
    pushTest("no_http_images_in_chat_before_image_prompt", httpBeforeImg === 0, `count=${httpBeforeImg}`);

    await page.getByTestId("grovee-prompt").fill("צור תמונה של חתול כחול");
    const tImg0 = Date.now();
    await page.getByTestId("grovee-send").click();
    await page.waitForFunction(
      () =>
        Array.from(document.querySelectorAll(".bubble img, .msg-body img")).some((img) =>
          (img.getAttribute("src") || "").startsWith("blob:"),
        ),
      null,
      { timeout: IMAGE_MS },
    );
    report.timingsMs.imageRequestToBlobVisible = Date.now() - tImg0;

    const httpAfter = await countChatHttpImages(page);
    const blobAfter = await countChatBlobImages(page);
    pushTest("local_blob_image_present", blobAfter > 0, `blobImages=${blobAfter}`);
    pushTest("no_http_https_img_src_in_chat", httpAfter === 0, `httpImages=${httpAfter}`);

    /** ניקוי שוב אחרי טעינה — אמור להקטין מטמון יחסית לנקודת אחרי־מודלים */
    await page.getByTestId("grovee-clear-cache").first().click();
    await page.getByText(/המטמון נוקה|ניקוי מטמון נכשל/).waitFor({ timeout: 120_000 });
    const snap2 = await storageSnapshot(page);
    report.storage.afterSecondClear = snap2;
    pushTest(
      "second_clear_reduces_or_maintains_modelish_caches",
      snap2.modelishCacheCount <= snapLoaded.modelishCacheCount,
      `loaded=${snapLoaded.modelishCacheCount} after2nd=${snap2.modelishCacheCount}`,
    );

    await page.screenshot({ path: path.join(OUT_DIR, "e2e-full.png"), fullPage: true });
    report.pageErrors = pageErrors.slice(0, 12);
    pushTest("no_page_errors", pageErrors.length === 0, pageErrors.slice(0, 3).join(" | "));
  } catch (e) {
    report.pass = false;
    report.fatal = e instanceof Error ? e.message : String(e);
    try {
      await page.screenshot({ path: path.join(OUT_DIR, "e2e-fatal.png"), fullPage: true });
    } catch {
      /* ignore */
    }
  } finally {
    report.timingsMs.total = Date.now() - tRun0;
    await browser.close();
  }

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  process.exit(report.pass ? 0 : 2);
}

run().catch((err) => {
  process.stderr.write(`${err?.stack || err}\n`);
  process.exit(1);
});
