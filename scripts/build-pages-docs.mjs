#!/usr/bin/env node
/** Build dist/ for GitHub Pages when the site is served from /docs/ on main. */
import { execSync } from "node:child_process";

process.env.VITE_BASE = "/GROVEEMODEL/docs/";
execSync("npm run build", { stdio: "inherit", env: process.env });
