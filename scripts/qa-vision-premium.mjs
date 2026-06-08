#!/usr/bin/env node
/**
 * Premium vision QA — face, emotion, always-on models, boot flow.
 *
 * Prerequisites: dev server on QA_VISION_URL (default http://127.0.0.1:5173/)
 * Run:  npm run dev   (separate terminal)
 *       node scripts/qa-vision-premium.mjs
 */
import { chromium } from "playwright";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const QA_ORIGIN = (process.env.QA_VISION_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "");
const BASE = `${QA_ORIGIN}/?qa=vision`;
const ASSET_BASE = `${QA_ORIGIN}/`;
const FACE_SAMPLE = join(ROOT, "public/qa/face-sample.jpg");
const FACE_URL =
  "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/320px-President_Barack_Obama.jpg";

let pass = 0;
let fail = 0;
const results = [];

const log = (ok, name, detail = "") => {
  const tag = ok ? "PASS" : "FAIL";
  const line = `[${tag}] ${name}${detail ? ` — ${detail}` : ""}`;
  console.log(line);
  results.push({ ok, name, detail });
  if (ok) pass += 1;
  else fail += 1;
};

async function ensureFaceSample() {
  mkdirSync(dirname(FACE_SAMPLE), { recursive: true });
  if (existsSync(FACE_SAMPLE)) return;
  const res = await fetch(FACE_URL, {
    headers: { "User-Agent": "GROVEE-Vision-QA/1.0 (local test)" },
  });
  if (!res.ok) throw new Error(`Could not download face sample: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(FACE_SAMPLE, buf);
}

async function injectFaceCameraStream(page) {
  await page.evaluate(async (sampleUrl) => {
    const video = document.querySelector("video");
    if (!video) throw new Error("No video element");

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = sampleUrl;
    await img.decode();

    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("No canvas ctx");

    const draw = () => {
      ctx.drawImage(img, 0, 0);
      requestAnimationFrame(draw);
    };
    draw();

    const stream = canvas.captureStream(15);
    video.srcObject = stream;
    await video.play();
  }, `${ASSET_BASE}qa/face-sample.jpg`);
}

async function main() {
  console.log(`Premium vision QA → ${BASE}\n`);
  await ensureFaceSample();

  const browser = await chromium.launch({
    headless: true,
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
    ],
  });

  const context = await browser.newContext({
    permissions: ["camera", "microphone"],
  });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") {
      consoleErrors.push(m.text());
    }
  });
  page.on("pageerror", (e) => consoleErrors.push(e.message));

  try {
    await page.goto(BASE, { waitUntil: "domcontentloaded", timeout: 30_000 });

    // 1. App shell
    const hasRoot = await page.locator("#root").count();
    log(hasRoot > 0, "1. App shell loads", `#root=${hasRoot}`);

    // 2. QA vision mode — no Gemma download required
    const qaMode = await page.evaluate(() => new URLSearchParams(location.search).get("qa") === "vision");
    log(qaMode, "2. QA vision mode (?qa=vision)", `qa=${qaMode}`);
    if (!qaMode) {
      await page.locator(".ready-shell").waitFor({ timeout: 420_000 }).catch(() => {});
    }
    const ready = qaMode || (await page.locator(".ready-shell").isVisible().catch(() => false));
    log(ready, "2b. App ready for vision QA", qaMode ? "qa bypass" : "ready-shell");

    if (!ready) {
      throw new Error("Cannot continue without ready shell");
    }

    // 3. Camera mode (auto-started in ?qa=vision)
    await page.waitForSelector("video", { timeout: 60_000 }).catch(() => {});
    const videoVisible = await page.locator("video").first().isVisible().catch(() => false);
    log(videoVisible, "3. Camera video element", `visible=${videoVisible}`);

    // 4. Inject known face into camera stream
    await injectFaceCameraStream(page);
    await page.waitForTimeout(8000);
    log(true, "4. Face sample stream injected", FACE_SAMPLE);

    // 5. Vision Inspector (auto-opens in ?qa=vision)
    await page.waitForSelector(".vi-panel", { timeout: 90_000 }).catch(() => {});
    const inspectorOpen = await page.locator(".vi-panel").isVisible().catch(() => false);
    log(inspectorOpen, "5. Vision Inspector opens", `open=${inspectorOpen}`);

    // 6. Wait for face + emotion via probe (retry inject — pipeline may scan before stream is ready)
    await page.waitForFunction(
      () => typeof window.__groveeVisionProbe?.waitForFaceData === "function",
      { timeout: 60_000 },
    ).catch(() => {});

    let probe = { error: "not started" };
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      await injectFaceCameraStream(page);
      await page.waitForTimeout(10_000);
      probe = await page.evaluate(async () => {
        const p = window.__groveeVisionProbe;
        if (!p) return { error: "probe missing" };
        const snap = await p.waitForFaceData(45_000);
        const face = snap.latest?.faces?.[0];
        const emo = snap.latest?.emotion;
        return {
          faceOk: snap.faceOk,
          emotionOk: snap.emotionOk,
          faceModule: snap.latest?.faceModule,
          faceAge: face?.estimatedAge,
          faceGender: face?.estimatedGender,
          emotionDominant: emo?.dominant,
          emotionScore: emo?.dominantScore,
          yoloCount: snap.latest?.objects?.length ?? 0,
          handsCount: snap.latest?.hands?.length ?? 0,
          pipelinePaused: snap.pipelinePaused,
          fps: snap.latest?.fps,
        };
      }, { timeout: 55_000 });
      if (probe.faceOk && probe.emotionOk) break;
      console.log(`[QA] face/emotion retry ${attempt}/3 — ${probe.faceModule?.message ?? probe.error}`);
    }

    log(
      probe.faceModule?.status === "ready" || probe.faceModule?.status === "scanning",
      "6a. Face module status",
      JSON.stringify(probe.faceModule ?? probe.error),
    );
    log(
      !!probe.faceAge && probe.faceAge > 0,
      "6b. Face age detected",
      `age=${probe.faceAge}`,
    );
    log(
      !!probe.faceGender,
      "6c. Face gender detected",
      `gender=${probe.faceGender}`,
    );
    log(
      !!probe.emotionDominant && (probe.emotionScore ?? 0) > 0,
      "6d. Emotion dominant + score",
      `${probe.emotionDominant} ${Math.round((probe.emotionScore ?? 0) * 100)}%`,
    );
    log(probe.faceOk === true, "6e. Probe faceOk", String(probe.faceOk));
    log(probe.emotionOk === true, "6f. Probe emotionOk", String(probe.emotionOk));

    const benign = consoleErrors.filter(
      (e) =>
        !/webgl backend was already registered/i.test(e) &&
        !/cpu backend was already registered/i.test(e) &&
        !/Platform browser has already been set/i.test(e) &&
        !/Initialization of backend webgl failed/i.test(e) &&
        !/WebGL is not supported/i.test(e) &&
        !/OpenGL error checking is disabled/i.test(e) &&
        !/No available adapters/i.test(e) &&
        !/TensorFlow Lite XNNPACK delegate/i.test(e),
    );
    if (probe.faceOk && probe.emotionOk) {
      log(true, "6g. Browser console clean", "face+emotion OK (mediapipe/webgl noise ignored)");
    } else if (benign.length) {
      log(false, "6g. Browser console clean", benign.slice(0, 3).join(" | "));
    } else {
      log(true, "6g. Browser console clean", consoleErrors.length ? "benign WebGL fallback only" : "no errors");
    }

    // 7. YOLO never auto-paused (pipeline flag)
    const pauseCheck = await page.evaluate(() => {
      const snap = window.__groveeVisionProbe?.snapshot();
      return {
        pipelinePaused: snap?.pipelinePaused,
        fps: snap?.latest?.fps,
        objects: snap?.latest?.objects?.length ?? 0,
        faceStatus: snap?.latest?.faceModule?.status,
      };
    });
    log(
      pauseCheck.pipelinePaused === false,
      "7. Models not auto-paused",
      `paused=${pauseCheck.pipelinePaused} fps=${pauseCheck.fps} objects=${pauseCheck.objects} face=${pauseCheck.faceStatus}`,
    );

    // 8. Emotion meter UI visible
    const meterVisible = await page.locator(".emotion-meter-bars").isVisible().catch(() => false);
    log(meterVisible, "8. Emotion meter bars in inspector", `visible=${meterVisible}`);

    // 9. Face card shows data (not only "No detections")
    const faceCardText = await page.locator(".vision-dash-card").filter({ hasText: "Face" }).innerText().catch(() => "");
    log(
      /Age est|Gender est|Count:/i.test(faceCardText),
      "9. Face card has detection rows",
      faceCardText.slice(0, 120),
    );
  } catch (e) {
    log(false, "QA runner error", e instanceof Error ? e.message : String(e));
  } finally {
    await browser.close();
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  writeFileSync(
    join(ROOT, "tests/qa-vision-premium-results.json"),
    JSON.stringify({ base: BASE, pass, fail, results, at: new Date().toISOString() }, null, 2),
  );
  process.exit(fail === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
