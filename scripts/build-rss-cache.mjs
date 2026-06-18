#!/usr/bin/env node
/**
 * Build-time RSS cache for GitHub Pages (no browser CORS / relay flooding).
 * Writes rss-cache.json into dist/ (and optionally docs/).
 *
 * Usage: node --import tsx scripts/build-rss-cache.mjs [--out=dist]
 */
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ALL_CATALOG_FEEDS } from "../app/src/groveeNews/engine/feeds/catalog/index.ts";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const outArg = process.argv.find((a) => a.startsWith("--out="))?.split("=")[1] ?? "dist";
const outDir = path.join(root, outArg);

const BATCH = 4;
const BATCH_DELAY_MS = 400;
const TIMEOUT_MS = 18_000;
const UA = "GROVEEMODEL-RSS-Cache/1.0 (+https://github.com/Theicd/GROVEEMODEL)";

function isRssXml(text) {
  return /<rss[\s>]/i.test(text) || /<feed[\s>]/i.test(text) || /<item[\s>]/i.test(text) || /<entry[\s>]/i.test(text);
}

async function fetchFeedXml(feed) {
  const urls = [feed.url, ...(feed.fallbackUrls ?? [])];
  for (const url of urls) {
    try {
      const res = await fetch(url, {
        headers: {
          Accept: "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
          "User-Agent": UA,
        },
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      if (!res.ok) continue;
      const xml = await res.text();
      if (!xml.trim() || !isRssXml(xml)) continue;
      return { url, xml };
    } catch {
      /* try next URL */
    }
  }
  return null;
}

async function runBatch(feeds) {
  return Promise.all(
    feeds.map(async (feed) => {
      const hit = await fetchFeedXml(feed);
      return { feed, hit };
    }),
  );
}

async function main() {
  const feeds = ALL_CATALOG_FEEDS;
  console.log(`[build-rss-cache] Fetching ${feeds.length} feeds (batch=${BATCH})…`);

  const byKey = {};
  const byUrl = {};
  let okCount = 0;

  for (let i = 0; i < feeds.length; i += BATCH) {
    const chunk = feeds.slice(i, i + BATCH);
    const results = await runBatch(chunk);
    for (const { feed, hit } of results) {
      if (hit) {
        okCount += 1;
        byKey[feed.key] = { url: hit.url, xml: hit.xml };
        byUrl[hit.url] = hit.xml;
        if (feed.url !== hit.url) byUrl[feed.url] = hit.xml;
        for (const fb of feed.fallbackUrls ?? []) {
          if (!byUrl[fb]) byUrl[fb] = hit.xml;
        }
        console.log(`  OK  ${feed.key} (${hit.xml.length}b)`);
      } else {
        byKey[feed.key] = { url: feed.url, ok: false };
        console.log(`  FAIL ${feed.key}`);
      }
    }
    if (i + BATCH < feeds.length) {
      await new Promise((r) => setTimeout(r, BATCH_DELAY_MS));
    }
  }

  const payload = {
    generatedAt: new Date().toISOString(),
    feedCount: feeds.length,
    okCount,
    byKey,
    byUrl,
  };

  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "rss-cache.json");
  writeFileSync(outPath, JSON.stringify(payload), "utf8");
  const sizeMb = (Buffer.byteLength(JSON.stringify(payload)) / (1024 * 1024)).toFixed(2);
  console.log(`[build-rss-cache] Wrote ${outPath} — ${okCount}/${feeds.length} feeds (${sizeMb} MB)`);

  if (okCount < Math.min(8, Math.floor(feeds.length * 0.1))) {
    console.error("[build-rss-cache] Too few feeds succeeded — check network or feed URLs.");
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("[build-rss-cache]", err);
  process.exit(1);
});
