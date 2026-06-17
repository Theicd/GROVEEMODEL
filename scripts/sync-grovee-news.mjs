/**
 * Sync GROVEE-NEWS engine sources into GROVEEMODEL/app/src/groveeNews/engine/
 * Run before dev/build: npm run sync:news
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_ROOT = path.resolve(__dirname, "..");
const NEWS_ROOT = process.env.GROVEE_NEWS_ROOT || path.resolve(MODEL_ROOT, "..", "GROVEE-NEWS");
const SRC = path.join(NEWS_ROOT, "src");
const DEST = path.join(MODEL_ROOT, "app", "src", "groveeNews", "engine");
const OVERLAY = path.join(MODEL_ROOT, "app", "src", "groveeNews", "engine-overlays");

/** UI-only — not part of the embedded engine. */
const SKIP_NAMES = new Set(["components", "App.tsx", "main.tsx"]);

function copyDir(srcDir, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  for (const name of fs.readdirSync(srcDir)) {
    if (SKIP_NAMES.has(name)) continue;
    const src = path.join(srcDir, name);
    const dest = path.join(destDir, name);
    const stat = fs.statSync(src);
    if (stat.isDirectory()) {
      copyDir(src, dest);
    } else if (stat.isFile()) {
      fs.copyFileSync(src, dest);
    }
  }
}

function applyOverlays() {
  if (!fs.existsSync(OVERLAY)) return;
  let n = 0;
  function walk(rel = "") {
    const srcDir = path.join(OVERLAY, rel);
    for (const name of fs.readdirSync(srcDir)) {
      const src = path.join(srcDir, name);
      const dest = path.join(DEST, rel, name);
      const stat = fs.statSync(src);
      if (stat.isDirectory()) {
        walk(path.join(rel, name));
      } else if (stat.isFile()) {
        fs.mkdirSync(path.dirname(dest), { recursive: true });
        fs.copyFileSync(src, dest);
        n += 1;
      }
    }
  }
  walk();
  if (n > 0) console.log(`applied ${n} engine overlay file(s) from engine-overlays/`);
}

if (!fs.existsSync(SRC)) {
  console.error(`GROVEE-NEWS src not found: ${SRC}`);
  console.error("Set GROVEE_NEWS_ROOT or place GROVEE-NEWS next to GROVEEMODEL.");
  process.exit(1);
}

if (fs.existsSync(DEST)) {
  fs.rmSync(DEST, { recursive: true, force: true });
}
copyDir(SRC, DEST);
applyOverlays();

const count = (dir) => {
  let n = 0;
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    if (fs.statSync(p).isDirectory()) n += count(p);
    else n += 1;
  }
  return n;
};

console.log(`sync-grovee-news: ${count(DEST)} files → ${path.relative(MODEL_ROOT, DEST)}`);

const pipelinePath = path.join(DEST, "engine", "pipeline.ts");
if (fs.existsSync(pipelinePath)) {
  let src = fs.readFileSync(pipelinePath, "utf8");
  if (!src.includes('from "./deepReadGate"')) {
    src = src.replace(
      'import { getEngineLibraryStats } from "./engineStats";',
      'import { getEngineLibraryStats } from "./engineStats";\nimport { isDeepReadEnabled } from "./deepReadGate";',
    );
    fs.writeFileSync(pipelinePath, src);
    console.log("patched pipeline.ts: isDeepReadEnabled import");
  }
}

const flexPath = path.join(DEST, "search", "flexIndex.ts");
if (fs.existsSync(flexPath)) {
  let src = fs.readFileSync(flexPath, "utf8");
  if (src.includes("RawSearchHit") && !src.includes("type RawSearchHit")) {
    src = src.replace(
      'import type { ArticleRecord } from "../types";',
      'import type { ArticleRecord } from "../types";\n\ntype RawSearchHit = { id: string; score: number };',
    );
    fs.writeFileSync(flexPath, src);
    console.log("patched flexIndex.ts: RawSearchHit type");
  }
}

function addNoCheck(dir) {
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const stat = fs.statSync(p);
    if (stat.isDirectory()) {
      addNoCheck(p);
      continue;
    }
    if (!name.endsWith(".ts") || name.endsWith(".test.ts")) continue;
    let src = fs.readFileSync(p, "utf8");
    if (!src.startsWith("// @ts-nocheck")) {
      fs.writeFileSync(p, `// @ts-nocheck\n${src}`);
    }
  }
}
addNoCheck(DEST);
console.log("applied @ts-nocheck to vendored engine sources");
