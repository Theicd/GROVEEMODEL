#!/usr/bin/env node
/**
 * Download face-api.js weight shards into public/models/face-api/.
 * Manifests are committed; shards are fetched on postinstall / npm run models:face.
 */
import { createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const OUT_DIR = join(ROOT, "public", "models", "face-api");

const CDN_BASE =
  "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";

const MANIFESTS = [
  "tiny_face_detector_model-weights_manifest.json",
  "face_landmark_68_model-weights_manifest.json",
  "face_expression_model-weights_manifest.json",
  "age_gender_model-weights_manifest.json",
];

const collectShardPaths = () => {
  const shards = new Set();
  for (const manifest of MANIFESTS) {
    const manifestPath = join(OUT_DIR, manifest);
    if (!existsSync(manifestPath)) continue;
    try {
      const parsed = JSON.parse(readFileSync(manifestPath, "utf8"));
      for (const block of parsed) {
        for (const p of block.paths ?? []) shards.add(p);
      }
    } catch {
      // fall through to defaults below
    }
  }
  if (shards.size === 0) {
    for (const name of [
      "tiny_face_detector_model-shard1",
      "face_landmark_68_model-shard1",
      "face_expression_model-shard1",
      "age_gender_model-shard1",
    ]) {
      shards.add(name);
    }
  }
  return [...shards];
};

const downloadFile = async (url, dest) => {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for ${url}`);
  }
  if (!res.body) {
    throw new Error(`Empty body for ${url}`);
  }
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
};

const main = async () => {
  mkdirSync(OUT_DIR, { recursive: true });

  const shards = collectShardPaths();
  const missing = shards.filter((name) => !existsSync(join(OUT_DIR, name)));

  if (missing.length === 0) {
    console.log("[GROVEE] Face models OK:", OUT_DIR);
    const listed = readdirSync(OUT_DIR).filter((f) => f.includes("shard"));
    console.log(`  shards present: ${listed.length}`);
    return;
  }

  console.log(`[GROVEE] Downloading ${missing.length} face-api shard(s)…`);

  for (const name of missing) {
    const url = `${CDN_BASE}/${name}`;
    const dest = join(OUT_DIR, name);
    process.stdout.write(`  → ${name} … `);
    try {
      await downloadFile(url, dest);
      console.log("OK");
    } catch (err) {
      console.log("FAILED");
      console.error(`[GROVEE] ${err instanceof Error ? err.message : err}`);
      console.error(
        "[GROVEE] Retry: npm run models:face  (or check network / CDN access)",
      );
      process.exit(1);
    }
  }

  console.log("[GROVEE] Face models ready:", OUT_DIR);
};

main().catch((err) => {
  console.error("[GROVEE] download-face-models failed:", err);
  process.exit(1);
});
