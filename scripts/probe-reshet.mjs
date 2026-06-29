import { gunzipSync } from "node:zlib";

function normalizeForMatch(raw) {
  return raw
    .replace(/\s*\(\d+p\)\s*/gi, " ")
    .replace(/\s*\[[^\]]*\]\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\u0590-\u05ff]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseChannels(xml) {
  const out = [];
  const re = /<channel id="([^"]+)"[^>]*>\s*<display-name>([^<]*)<\/display-name>/g;
  let m;
  while ((m = re.exec(xml))) out.push({ id: m[1], name: m[2] });
  return out;
}

function matchChannel(channels, title, matchKey) {
  const key = matchKey.trim().toLowerCase();
  const exact = channels.find((c) => c.name.toLowerCase() === key || c.id.toLowerCase() === key);
  if (exact) return exact;
  const contains = channels.filter(
    (c) => c.name.toLowerCase().includes(key) || key.includes(c.name.toLowerCase()),
  );
  return contains[0] ?? null;
}

const res = await fetch("https://i.mjh.nz/all/epg.xml.gz");
const xml = gunzipSync(Buffer.from(await res.arrayBuffer())).toString("utf8");
const channels = parseChannels(xml);
const title = "Reshet 13 (720p)";
const keys = [normalizeForMatch(title), "channel13.il", "reshet 13", "channel 13"];
for (const key of keys) {
  const ch = matchChannel(channels, title, key);
  if (ch) console.log("MATCH", key, "->", ch.id, "|", ch.name);
}
