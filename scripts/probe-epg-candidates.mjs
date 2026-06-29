/**
 * Suggest EPG channel matches for NO_MATCH favorites.
 * Run: node scripts/probe-epg-candidates.mjs
 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const favorites = JSON.parse(readFileSync(join(root, "public/liveMedia/curatedFavorites.json"), "utf8"));
const results = JSON.parse(readFileSync(join(root, "scripts/probe-all-favorites-results.json"), "utf8"));
const noMatch = results.filter((r) => r.status === "NO_MATCH").map((r) => r.name);

const SOURCES = [
  { key: "plex", url: "https://i.mjh.nz/Plex/us.xml.gz" },
  { key: "pluto", url: "https://i.mjh.nz/PlutoTV/us.xml.gz" },
  { key: "samsung", url: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz" },
  { key: "roku", url: "https://i.mjh.nz/Roku/all.xml.gz" },
  { key: "all", url: "https://i.mjh.nz/all/epg.xml.gz" },
];

function parseChannels(xml) {
  const out = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  while ((m = re.exec(xml))) out.push({ id: m[1], name: m[2] });
  return out;
}

function tokens(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((t) => t.length >= 3);
}

function compact(s) {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

const xmlByKey = new Map();
for (const src of SOURCES) {
  const buf = Buffer.from(await (await fetch(src.url)).arrayBuffer());
  xmlByKey.set(src.key, gunzipSync(buf).toString("utf8"));
}

for (const name of noMatch) {
  const ch = favorites.channels.find((c) => c.name === name);
  if (!ch) {
    console.log("MISSING", name);
    continue;
  }
  const title = ch.name.replace(/\(\d+p\)/gi, "").replace(/\[[^\]]*\]/g, "").trim();
  const tvg = ch.tvgId || "";
  const toks = new Set([...tokens(title), ...tokens(tvg.split("@")[0] || "")]);
  const streamToks = ch.stream.match(/[a-z0-9]{4,}/gi) || [];
  for (const st of streamToks) if (st.length >= 5) toks.add(st.toLowerCase());

  console.log(`\n=== ${name} ===`);
  console.log(`  tvgId: ${tvg}`);
  console.log(`  stream: ${ch.stream}`);

  const hits = [];
  for (const [key, xml] of xmlByKey) {
    for (const c of parseChannels(xml)) {
      const cn = c.name.toLowerCase();
      const id = c.id.toLowerCase();
      const idCompact = compact(c.id);
      let score = 0;
      for (const t of toks) {
        if (cn.includes(t)) score += 2;
        const tc = compact(t);
        if (tc.length >= 4 && idCompact.includes(tc)) score += 2;
      }
      const tvgBase = compact(tvg.split("@")[0] || "");
      if (tvgBase.length >= 4 && (idCompact.includes(tvgBase) || compact(cn).includes(tvgBase))) score += 12;
      if (score >= 4) hits.push({ score, src: key, name: c.name, id: c.id });
    }
  }
  hits.sort((a, b) => b.score - a.score);
  for (const h of hits.slice(0, 6)) {
    console.log(`  ${h.score} ${h.src.padEnd(7)} ${h.name} | ${h.id.slice(0, 48)}`);
  }
}
