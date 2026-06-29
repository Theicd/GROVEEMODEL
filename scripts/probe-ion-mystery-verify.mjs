/**
 * Verify ION Mystery / WFXT-DT2 EPG match against live MJH Plex feed.
 */
import { gunzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dir = dirname(fileURLToPath(import.meta.url));
const root = join(__dir, "..");

// Inline minimal copies of matching logic (keep in sync with app)
const GENERIC = new Set(["on", "the", "and", "hd", "live", "tv"]);
const IGNORE = new Set(["master", "playlist", "index", "live", "stream", "hls", "m3u8"]);

function sig(s) {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length >= 2 && !GENERIC.has(t));
}
function compact(s) {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}
function pathTokens(url) {
  const tokens = new Set();
  for (const seg of new URL(url).pathname.split("/").filter(Boolean)) {
    for (const t of sig(seg.replace(/\.m3u8$/i, "").replace(/_/g, " "))) {
      if (!IGNORE.has(t)) tokens.add(t);
    }
  }
  return [...tokens];
}
function score(ch, title, stream) {
  const nc = compact(ch.name);
  const pc = compact(new URL(stream).pathname);
  if (nc.length >= 6 && pc.includes(nc)) return 86;
  const ts = sig(title.replace(/\(\d+p\)/gi, ""));
  if (ts.length >= 2 && ts.every((t) => sig(ch.name).includes(t))) return 90;
  if (compact(title) === nc) return 100;
  return 0;
}

const fav = JSON.parse(readFileSync(join(root, "public/liveMedia/curatedFavorites.json"), "utf8"));
const ch = fav.channels.find((c) => c.id === "2vwcmz");
if (!ch) throw new Error("favorite 2vwcmz missing");

const res = await fetch("https://i.mjh.nz/Plex/us.xml.gz");
const xml = gunzipSync(Buffer.from(await res.arrayBuffer())).toString("utf8");
const channels = [];
const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/gi;
let m;
while ((m = re.exec(xml))) channels.push({ id: m[1], name: m[2] });

const titles = ["ION Mystery", "Ion Mystery", "WFXT-DT2", ch.name.replace(/\(\d+p\)/, "").trim()];
let best = null;
for (const t of titles) {
  for (const c of channels) {
    const s = score(c, t, ch.stream);
    if (s >= 65 && (!best || s > best.score)) best = { c, t, score: s };
  }
}

console.log("Channel:", ch.name);
console.log("Stream:", ch.stream);
if (!best) {
  console.error("NO_MATCH");
  process.exit(1);
}
console.log("MATCH:", best.c.name, "|", best.c.id, "| score", best.score, "| via title", best.t);

const progRe = new RegExp(
  `<programme\\s[^>]*channel="${best.c.id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"[^>]*start="([^"]+)"[^>]*stop="([^"]+)"[^>]*>[\\s\\S]*?<title>([^<]*)<`,
  "i",
);
const pm = progRe.exec(xml);
console.log("Sample programme:", pm ? pm[3] : "none");
