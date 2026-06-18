#!/usr/bin/env node
/**
 * QA: unified SERP panel — merge layer + routing + provider formatting.
 * Run: npm run qa:unified-search
 */
import { spawnSync } from "node:child_process";

const steps = [
  ["vitest", "run", "app/src/searchResults/mergeSearchHits.test.ts"],
  ["vitest", "run", "app/src/webSearch/searchBrief.github.test.ts"],
  ["vitest", "run", "app/src/webSearch/searchPlanner.test.ts"],
  ["vitest", "run", "app/src/groveeNews/newsToSearchBrief.test.ts"],
  ["vitest", "run", "app/src/webSearch/webSearch.test.ts"],
];

let failed = 0;
for (const args of steps) {
  const label = args.slice(2).join("/");
  console.log(`\n▶ ${label}`);
  const r = spawnSync("npx", args, { stdio: "inherit", shell: true });
  if (r.status !== 0) failed += 1;
}

if (failed) {
  console.error(`\n✗ unified-search QA: ${failed} step(s) failed`);
  process.exit(1);
}
console.log("\n✓ unified-search QA passed");
