/** Probe every NEWS_FEEDS entry — writes RSS_FEED_PROBE_REPORT.md */
import { writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { NEWS_FEEDS } from "../app/src/webSearch/providers/newsFeeds.ts";
import { parseRssTitles } from "../app/src/webSearch/providers/newsRss.ts";

const __dir = dirname(fileURLToPath(import.meta.url));
const REPORT = join(__dir, "..", "RSS_FEED_PROBE_REPORT.md");

const RELAYS = [
  (u: string) => `https://corsproxy.io/?${encodeURIComponent(u)}`,
  (u: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  (u: string) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(u)}`,
];

async function fetchFeed(url: string): Promise<{ ok: boolean; body: string; via: string; error?: string }> {
  const headers = { Accept: "application/rss+xml, application/xml, text/xml, */*" };
  try {
    const r = await fetch(url, { headers, signal: AbortSignal.timeout(12_000) });
    if (r.ok) {
      const body = await r.text();
      if (body.includes("<item") || body.includes("<entry")) return { ok: true, body, via: "direct" };
    }
  } catch (e) {
    /* try relays */
  }
  for (const relay of RELAYS) {
    try {
      const r = await fetch(relay(url), { headers, signal: AbortSignal.timeout(15_000) });
      if (!r.ok) continue;
      const body = await r.text();
      if (body.includes("<item") || body.includes("<entry")) return { ok: true, body, via: "relay" };
    } catch {
      /* next */
    }
  }
  return { ok: false, body: "", via: "none", error: "all fetches failed" };
}

type Row = { key: string; label: string; url: string; ok: boolean; via: string; titles: string[]; error?: string };

const rows: Row[] = [];
const keys = Object.keys(NEWS_FEEDS).sort();

for (let i = 0; i < keys.length; i++) {
  const key = keys[i]!;
  const feed = NEWS_FEEDS[key]!;
  process.stdout.write(`\r[${i + 1}/${keys.length}] ${key}…`);
  const urls = [feed.url, ...(feed.fallbackUrls ?? [])];
  let result: { ok: boolean; body: string; via: string; error?: string } = {
    ok: false,
    body: "",
    via: "none",
    error: "all fetches failed",
  };
  for (const url of urls) {
    result = await fetchFeed(url);
    if (result.ok) break;
  }
  const titles = result.ok ? parseRssTitles(result.body, 3) : [];
  rows.push({
    key,
    label: feed.label,
    url: feed.url,
    ok: result.ok && titles.length > 0,
    via: result.via,
    titles,
    error: result.ok && !titles.length ? "no titles parsed" : result.error,
  });
  await new Promise((r) => setTimeout(r, 150));
}
console.log("\n");

const ok = rows.filter((r) => r.ok);
const fail = rows.filter((r) => !r.ok);

let md = `# RSS Feed Probe — ${new Date().toISOString().slice(0, 19)}\n\n`;
md += `✅ ${ok.length}/${rows.length} OK · ❌ ${fail.length} failed\n\n`;
md += `## OK (${ok.length})\n\n| key | label | via | sample |\n|-----|-------|-----|--------|\n`;
for (const r of ok) {
  md += `| ${r.key} | ${r.label} | ${r.via} | ${(r.titles[0] ?? "").replace(/\|/g, "/").slice(0, 60)} |\n`;
}
md += `\n## FAIL (${fail.length})\n\n| key | label | url | error |\n|-----|-------|-----|-------|\n`;
for (const r of fail) {
  md += `| ${r.key} | ${r.label} | ${r.url.slice(0, 50)}… | ${r.error ?? "no titles"} |\n`;
}

writeFileSync(REPORT, md, "utf8");
console.log(`Report → ${REPORT}`);
console.log(`✅ ${ok.length}  ❌ ${fail.length}`);
process.exit(fail.length > keys.length / 2 ? 1 : 0);
