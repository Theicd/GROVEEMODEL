/**
 * Sync reality-core UI into public/reality/ for static GitHub Pages deploy.
 * Source: ../reality-core/ui (sibling repo) when present; otherwise keep committed copy.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const SRC = path.resolve(ROOT, "../reality-core/ui");
const DEST = path.join(ROOT, "public", "reality");

const FILES = [
  "israel.html",
  "israel.js",
  "embed-grovee.css",
  "embed-bridge.js",
  "sounds.js",
  "local-analysis.js",
  "api-registry.js",
  "api-health-check.js",
  "qa-system.js",
  "qa-diagnostics.js",
  "api-validator.js",
];

function copyFile(src, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
}

if (fs.existsSync(SRC)) {
  console.log(`[sync-reality] copying from ${SRC}`);
  for (const f of FILES) {
    const from = path.join(SRC, f);
    if (!fs.existsSync(from)) {
      console.warn(`[sync-reality] skip missing: ${f}`);
      continue;
    }
    copyFile(from, path.join(DEST, f));
  }
  console.log(`[sync-reality] ✓ ${FILES.length} files → public/reality/`);
} else if (fs.existsSync(path.join(DEST, "israel.html"))) {
  console.log("[sync-reality] using committed public/reality/ (no sibling reality-core)");
} else {
  console.error("[sync-reality] ERROR: no source and no public/reality/israel.html");
  process.exit(1);
}
