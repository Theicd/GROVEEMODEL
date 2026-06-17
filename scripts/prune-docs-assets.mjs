#!/usr/bin/env node
/**
 * Remove stale hashed files under docs/assets/ only.
 * Follows references from index.html through the JS/MJS import graph (workers, chunks).
 */
import { readFileSync, readdirSync, unlinkSync, existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const docs = path.join(root, "docs");
const assetsDir = path.join(docs, "assets");
const indexHtml = path.join(docs, "index.html");

if (!existsSync(indexHtml) || !existsSync(assetsDir)) {
  console.log("[prune-docs-assets] skip — no docs bundle yet");
  process.exit(0);
}

const REF_PATTERNS = [
  /assets\/([a-zA-Z0-9._-]+)/g,
  /["'](?:\.\/)?([a-zA-Z0-9._-]+\.(?:js|mjs|wasm))["']/g,
];

function scanText(text, referenced, pending) {
  for (const re of REF_PATTERNS) {
    re.lastIndex = 0;
    for (const m of text.matchAll(re)) {
      const name = m[1];
      if (referenced.has(name) || !existsSync(path.join(assetsDir, name))) continue;
      referenced.add(name);
      pending.push(name);
    }
  }
}

function collectReferenced() {
  const referenced = new Set();
  const pending = [];

  scanText(readFileSync(indexHtml, "utf8"), referenced, pending);

  while (pending.length) {
    const name = pending.shift();
    const filePath = path.join(assetsDir, name);
    if (!existsSync(filePath) || !statSync(filePath).isFile()) continue;
    if (!/\.(js|mjs|css)$/i.test(name)) continue;
    scanText(readFileSync(filePath, "utf8"), referenced, pending);
  }

  for (const name of readdirSync(assetsDir)) {
    if (name.endsWith(".wasm")) referenced.add(name);
  }

  return referenced;
}

const referenced = collectReferenced();

let removed = 0;
for (const name of readdirSync(assetsDir)) {
  if (referenced.has(name)) continue;
  unlinkSync(path.join(assetsDir, name));
  removed += 1;
}

console.log(`[prune-docs-assets] kept ${referenced.size} file(s), removed ${removed} stale asset(s)`);
