/** Probe Gravitas matching — run: node scripts/probe-gravitas-match.mjs */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// inline old vs new - import after we build channelMatch
const fav = {
  name: "Gravitas Movies (1080p)",
  tvgId: "GravitasMovies.us@SD",
};

async function loadChannels(url, sourceKey) {
  const res = await fetch(url);
  const xml = gunzipSync(Buffer.from(await res.arrayBuffer())).toString("utf8");
  const out = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/gi;
  let m;
  while ((m = re.exec(xml))) out.push({ id: m[1], name: m[2], sourceKey });
  return out;
}

// dynamic import of ts - use vitest instead
console.log("Use vitest for match tests");
