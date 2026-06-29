import { gunzipSync } from "node:zlib";

const TERMS = [
  "charge", "entertainment", "discover", "positiv", "rally", "deluxe", "autentic",
  "inwild", "tune", "hollywood", "space", "teen", "detective", "jerry", "cipher",
  "mystery", "whiplash", "absolute", "reality", "fashion", "groovy", "classique",
  "history hunt", "fnx", "kvcr", "chat show", "wild", "film det", "tom ", "30a tv",
  "amc ", "wetv", "wurl", "sinclair", "cinedigm", "rakuten", "amagi", "bozztv",
];

const SOURCES = [
  ["plex", "https://i.mjh.nz/Plex/us.xml.gz"],
  ["pluto", "https://i.mjh.nz/PlutoTV/us.xml.gz"],
  ["samsung", "https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],
  ["roku", "https://i.mjh.nz/Roku/all.xml.gz"],
  ["all", "https://i.mjh.nz/all/epg.xml.gz"],
];

function parseChannels(xml) {
  const out = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  while ((m = re.exec(xml))) out.push({ id: m[1], name: m[2] });
  return out;
}

for (const [key, url] of SOURCES) {
  const buf = Buffer.from(await (await fetch(url)).arrayBuffer());
  const channels = parseChannels(gunzipSync(buf).toString("utf8"));
  console.log(`\n=== ${key} ===`);
  for (const term of TERMS) {
    const hits = channels.filter((c) => c.name.toLowerCase().includes(term));
    if (hits.length) {
      console.log(`  [${term}] ${hits.map((h) => h.name).join(" | ")}`);
    }
  }
}
