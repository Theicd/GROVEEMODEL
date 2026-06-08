#!/usr/bin/env node
/**
 * Aligns with package.json engines.node (>=18.18.0).
 * This project uses Vite 5 — Node 20.17+ and 18.18+ are fine.
 */
const ver = process.versions.node;
const [maj, min] = ver.split(".").map((x) => Number(x));

const ok = maj > 18 || (maj === 18 && min >= 18);

if (!ok) {
  console.error("");
  console.error("[GROVEE] Unsupported Node.js version:", ver);
  console.error("  Need Node.js 18.18 or newer (see package.json engines)");
  console.error("  Download LTS: https://nodejs.org/");
  console.error("");
  process.exit(1);
}

if (maj === 20 && min < 19) {
  console.log(
    "[GROVEE] Node 20." +
      min +
      " is OK for this build (Vite 5). For future Vite 8, prefer Node 22 LTS.",
  );
}

if (maj === 21) {
  console.warn(
    "[GROVEE] Node 21.x shows EBADENGINE warnings from ESLint 10 — install works; prefer Node 22 LTS for CI.",
  );
}

process.exit(0);
