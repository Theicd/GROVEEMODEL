/**
 * Search XMLTV feeds for channel display-names matching NO_MATCH favorites.
 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const results = JSON.parse(readFileSync(join(root, "scripts/probe-all-favorites-results.json"), "utf8"));
const favorites = JSON.parse(readFileSync(join(root, "public/liveMedia/curatedFavorites.json"), "utf8"));
const noMatch = results.filter((r) => r.status === "NO_MATCH");

const QUERIES = noMatch.map((r) => {
  const ch = favorites.channels.find((c) => c.name === r.name);
  const clean = r.name
    .replace(/like Gecko\)[^,]*,\s*/gi, "")
    .replace(/\(\d+p\)/gi, "")
    .replace(/\[[^\]]*\]/g, "")
    .trim();
  return { name: r.name, clean, tvgId: ch?.tvgId ?? "", stream: ch?.stream ?? "" };
});

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

function words(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(" ").filter((w) => w.length >= 3);
}

for (const src of SOURCES) {
  const buf = Buffer.from(await (await fetch(src.url)).arrayBuffer());
  const channels = parseChannels(gunzipSync(buf).toString("utf8"));
  console.log(`\n######## ${src.key} (${channels.length} channels) ########`);
  for (const q of QUERIES) {
    const qw = words(q.clean);
    const tvgBase = (q.tvgId.split("@")[0] || "").toLowerCase();
    const matches = channels.filter((c) => {
      const cn = c.name.toLowerCase();
      const id = c.id.toLowerCase();
      if (qw.length >= 2 && qw.every((w) => cn.includes(w))) return true;
      if (tvgBase.length >= 5) {
        const compact = tvgBase.replace(/[^a-z0-9]/g, "");
        if (id.includes(compact) || cn.replace(/[^a-z0-9]/g, "").includes(compact)) return true;
      }
      return false;
    });
    if (matches.length) {
      console.log(`\n${q.name}:`);
      for (const m of matches.slice(0, 4)) console.log(`  ${m.name} | ${m.id}`);
    }
  }
}
