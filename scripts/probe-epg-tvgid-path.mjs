import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const favorites = JSON.parse(readFileSync(join(root, "public/liveMedia/curatedFavorites.json"), "utf8"));
const results = JSON.parse(readFileSync(join(root, "scripts/probe-all-favorites-results.json"), "utf8"));
const noMatch = results.filter((r) => r.status === "NO_MATCH");

const SOURCES = [
  ["mjh-plex-us", "https://i.mjh.nz/Plex/us.xml.gz"],
  ["mjh-pluto-us", "https://i.mjh.nz/PlutoTV/us.xml.gz"],
  ["mjh-samsung-us", "https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],
  ["mjh-roku", "https://i.mjh.nz/Roku/all.xml.gz"],
  ["mjh-all", "https://i.mjh.nz/all/epg.xml.gz"],
];

function compact(s) {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function parseXmltvDate(raw) {
  const m = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s, tz] = m;
  return new Date(`${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`);
}

function upcomingCount(xml, channelId) {
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`<programme\\s[^>]*channel="${escaped}"[^>]*stop="([^"]+)"`, "g");
  const horizon = Date.now() - 6 * 60 * 60 * 1000;
  let n = 0;
  let m;
  while ((m = re.exec(xml))) {
    const end = parseXmltvDate(m[1]);
    if (end && end.getTime() > horizon) n++;
  }
  return n;
}

const xmlByKey = new Map();
const channelsByKey = new Map();
for (const [key, url] of SOURCES) {
  const xml = gunzipSync(Buffer.from(await (await fetch(url)).arrayBuffer())).toString("utf8");
  xmlByKey.set(key, xml);
  const channels = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  while ((m = re.exec(xml))) channels.push({ id: m[1], name: m[2] });
  channelsByKey.set(key, channels);
}

for (const r of noMatch) {
  const ch = favorites.channels.find((c) => c.name === r.name);
  const tvgBase = compact((ch.tvgId || "").split("@")[0] || "");
  const pathCompact = compact(new URL(ch.stream).pathname);
  console.log(`\n=== ${r.name} ===`);
  if (!tvgBase && !pathCompact) continue;
  for (const [key, channels] of channelsByKey) {
    const xml = xmlByKey.get(key);
    for (const c of channels) {
      const idC = compact(c.id);
      const nameC = compact(c.name);
      let hit = false;
      if (tvgBase.length >= 5 && (idC.includes(tvgBase) || nameC.includes(tvgBase))) hit = true;
      if (!hit && pathCompact.length >= 8) {
        for (let i = 0; i <= pathCompact.length - 8; i++) {
          const sub = pathCompact.slice(i, i + 8);
          if (idC.includes(sub) || nameC.includes(sub)) {
            hit = true;
            break;
          }
        }
      }
      if (!hit) continue;
      const upcoming = upcomingCount(xml, c.id);
      if (upcoming > 0) console.log(`  ${key}: ${c.name} (${upcoming} upcoming) ${c.id.slice(0, 48)}`);
    }
  }
}
