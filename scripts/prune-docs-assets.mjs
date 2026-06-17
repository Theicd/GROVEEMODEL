#!/usr/bin/env node
/** Remove hashed assets under docs/ that index.html (and its imports) no longer reference. */
import { readFileSync, readdirSync, unlinkSync, existsSync } from "node:fs";
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

const html = readFileSync(indexHtml, "utf8");
const referenced = new Set();
for (const m of html.matchAll(/\.\/assets\/([a-zA-Z0-9._-]+)/g)) {
  referenced.add(m[1]);
}

let removed = 0;
for (const name of readdirSync(assetsDir)) {
  if (referenced.has(name)) continue;
  unlinkSync(path.join(assetsDir, name));
  removed += 1;
}

console.log(`[prune-docs-assets] kept ${referenced.size} files, removed ${removed} stale asset(s)`);
