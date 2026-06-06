#!/usr/bin/env node
/**
 * Vite 5+ dev tooling: Node 18.18+ recommended.
 * Node 21.x works with Vite 5; Node 20.19+ / 22.12+ for Vite 8+.
 */
const ver = process.versions.node;
const [maj, min, patch] = ver.split(".").map((x) => Number(x));

const ok =
  (maj === 18 && min >= 18) ||
  (maj === 20 && min >= 19) ||
  maj === 21 ||
  (maj === 22 && min >= 12) ||
  maj >= 24;

if (!ok) {
  console.error("");
  console.error("[GROVEE] Unsupported Node.js version:", ver);
  console.error("  Need Node 18.18+, 20.19+, 21.x, 22.12+, or 24+");
  console.error("  Download LTS: https://nodejs.org/");
  console.error("");
  process.exit(1);
}

if (maj === 21) {
  console.log("[GROVEE] Node 21 detected — using Vite 5 (compatible). For Vite 8 upgrade to Node 22 LTS.");
}

process.exit(0);
