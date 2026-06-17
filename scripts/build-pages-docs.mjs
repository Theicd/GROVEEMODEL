#!/usr/bin/env node
/** Build dist/ for GitHub Pages when the site is served from /docs/ on main. */
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { execSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

// Relative base — assets load as ./assets/... from docs/index.html (see README).
process.env.VITE_BASE = "./";
execSync("npm run build", { stdio: "inherit", env: process.env, cwd: root });

const dist = path.join(root, "dist");
const docs = path.join(root, "docs");
if (existsSync(docs)) rmSync(docs, { recursive: true, force: true });
mkdirSync(docs, { recursive: true });
cpSync(dist, docs, { recursive: true, force: true });

execSync("node scripts/prune-docs-assets.mjs", { stdio: "inherit", cwd: root });

console.log("[build-pages-docs] synced dist/ → docs/ (clean, single bundle)");
