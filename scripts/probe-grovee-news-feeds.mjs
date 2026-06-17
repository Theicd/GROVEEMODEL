/**
 * Probe a sample of GROVEE-NEWS engine catalog feeds via dev /api/fetch proxy.
 * Run with dev server up: npm run dev then npm run qa:rss-feeds
 */
import { FEED_BY_KEY } from "../app/src/groveeNews/engine/feeds/feedRegistry.ts";

const SAMPLE_KEYS = ["bbc", "ynet", "reuters", "guardian", "ap"];
const base = process.env.GROVEE_DEV_URL || "http://127.0.0.1:5180";

async function probeFeed(key, url) {
  const proxy = `${base}/api/fetch?url=${encodeURIComponent(url)}`;
  const res = await fetch(proxy, { signal: AbortSignal.timeout(20_000) });
  const text = await res.text();
  const ok = res.ok && (text.includes("<item") || text.includes("<entry"));
  console.log(`${ok ? "OK" : "FAIL"} ${key} ${res.status} ${text.length}b`);
  return ok;
}

let ok = 0;
for (const key of SAMPLE_KEYS) {
  const feed = FEED_BY_KEY[key];
  if (!feed) {
    console.log(`SKIP ${key} (missing)`);
    continue;
  }
  if (await probeFeed(key, feed.url)) ok += 1;
}
console.log(`\n${ok}/${SAMPLE_KEYS.length} sample feeds responded`);
process.exit(ok > 0 ? 0 : 1);
