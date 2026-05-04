#!/usr/bin/env node
/**
 * Live deployment health-check for https://theicd.github.io/GROVEEMODEL/.
 *
 * Steps (each is a separate, named test):
 *   1. GET / on Pages → 200, has <div id="root">.
 *   2. Parse <script type="module" src=...> + <link rel="stylesheet" href=...> from index.html
 *      and HEAD-fetch each → 200 with the expected content-type.
 *   3. HEAD-fetch a known Pollinations image URL built by app code → 200 + image/* mime.
 *   4. Plain GET of that Pollinations URL streams back image bytes (>= 4 magic bytes).
 *
 * Run:   node scripts/qa-pages.mjs
 *        node scripts/qa-pages.mjs --base=https://theicd.github.io/GROVEEMODEL/
 *
 * CI:    set $env:QA_PAGES_BASE = "https://theicd.github.io/GROVEEMODEL/"
 *        npm run qa:pages
 */

import { buildPollinationsUrl } from "../app/src/cloudImage.ts";

const argBase = process.argv.find((a) => a.startsWith("--base="))?.split("=")[1];
const BASE = (argBase ?? process.env.QA_PAGES_BASE ?? "https://theicd.github.io/GROVEEMODEL/").replace(/\/?$/, "/");

let pass = 0;
let fail = 0;
const results = [];

const log = (status, name, detail = "") => {
  const tag = status === "ok" ? "PASS" : status === "skip" ? "SKIP" : "FAIL";
  results.push({ status, name, detail });
  const line = `[${tag}] ${name}${detail ? ` — ${detail}` : ""}`;
  if (status === "ok") {
    console.log(line);
    pass += 1;
  } else if (status === "skip") {
    console.log(line);
  } else {
    console.error(line);
    fail += 1;
  }
};

const FETCH_TIMEOUT_MS = 25_000;

async function fetchWithTimeout(url, opts = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, { ...opts, signal: ctrl.signal });
  } finally {
    clearTimeout(t);
  }
}

async function step(name, fn) {
  try {
    const detail = await fn();
    log("ok", name, detail ?? "");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    log("fail", name, msg);
  }
}

async function main() {
  console.log(`QA target: ${BASE}\n`);

  let html = "";
  let appBase = BASE;
  await step("1. GET app HTML (follow meta-refresh) → 200 with #root", async () => {
    const fetchHtml = async (url) => {
      const r = await fetchWithTimeout(url);
      if (r.status !== 200) throw new Error(`HTTP ${r.status} on ${url}`);
      return [await r.text(), url];
    };
    let url = BASE;
    for (let hop = 0; hop < 3; hop += 1) {
      const [text, finalUrl] = await fetchHtml(url);
      html = text;
      appBase = finalUrl;
      if (/id=["']root["']/.test(html)) return `${html.length} bytes (${appBase})`;
      const meta = html.match(/<meta[^>]+http-equiv=["']refresh["'][^>]+content=["']\s*\d+\s*;\s*url=([^"']+)["']/i);
      const jsLoc = html.match(/location\.replace\(\s*new URL\(\s*["']([^"']+)["']/i);
      const next = meta?.[1] ?? jsLoc?.[1];
      if (!next) throw new Error("Missing #root and no meta-refresh / location.replace target");
      url = new URL(next, url).toString();
    }
    throw new Error("too many redirects");
  });

  const linkRefs = [];
  if (html) {
    const scriptRe = /<script[^>]+src=["']([^"']+)["']/gi;
    const linkRe = /<link[^>]+href=["']([^"']+)["'][^>]*rel=["'](?:stylesheet|modulepreload)["']/gi;
    const altLinkRe = /<link[^>]+rel=["'](?:stylesheet|modulepreload)["'][^>]+href=["']([^"']+)["']/gi;
    for (const m of html.matchAll(scriptRe)) linkRefs.push({ kind: "script", url: new URL(m[1], appBase).toString() });
    for (const m of html.matchAll(linkRe)) linkRefs.push({ kind: "link", url: new URL(m[1], appBase).toString() });
    for (const m of html.matchAll(altLinkRe)) linkRefs.push({ kind: "link", url: new URL(m[1], appBase).toString() });
  }
  if (linkRefs.length === 0) {
    log("fail", "2. parse asset URLs", "no <script>/<link> found in index.html");
  } else {
    for (const ref of linkRefs) {
      await step(`2. HEAD ${ref.kind} ${ref.url}`, async () => {
        const r = await fetchWithTimeout(ref.url, { method: "GET", headers: { Range: "bytes=0-0" } });
        if (r.status !== 200 && r.status !== 206) throw new Error(`HTTP ${r.status}`);
        const ct = r.headers.get("content-type") ?? "";
        const wantsJs = ref.kind === "script" || ref.url.endsWith(".js");
        const wantsCss = ref.url.endsWith(".css");
        if (wantsJs && !/javascript|ecmascript|module/i.test(ct)) throw new Error(`unexpected content-type: ${ct}`);
        if (wantsCss && !/css/i.test(ct)) throw new Error(`unexpected content-type: ${ct}`);
        return ct;
      });
    }
  }

  const cloud = buildPollinationsUrl({ prompt: "a tiny red cube on a desk, soft light, photorealistic" });
  await step("3. HEAD Pollinations image URL → 200 + image/*", async () => {
    const r = await fetchWithTimeout(cloud, { method: "GET", headers: { Range: "bytes=0-0" } });
    if (r.status !== 200 && r.status !== 206) throw new Error(`HTTP ${r.status}`);
    const ct = r.headers.get("content-type") ?? "";
    if (!/^image\//i.test(ct)) throw new Error(`unexpected content-type: ${ct}`);
    return ct;
  });

  await step("4. GET Pollinations image returns image bytes", async () => {
    const r = await fetchWithTimeout(cloud);
    if (r.status !== 200) throw new Error(`HTTP ${r.status}`);
    const buf = new Uint8Array(await r.arrayBuffer());
    if (buf.byteLength < 32) throw new Error(`tiny body: ${buf.byteLength} bytes`);
    const magic = String.fromCharCode(...buf.slice(0, 4));
    return `${buf.byteLength} bytes, magic=${JSON.stringify(magic)}`;
  });

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("unexpected:", e);
  process.exit(2);
});
