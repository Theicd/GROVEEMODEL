import { gunzipSync } from "node:zlib";

const TERMS = [
  "30a", "abc kids", "absolute reality", "autentic", "charge", "classique", "deluxe music",
  "discoverfilm", "discover film", "entertainment tonight", "fifa", "fnx", "kvcr",
  "global fashion", "groovy", "history hunters", "inwild", "tunebox", "360tune",
  "mix hollywood", "positiv", "rally tv", "space series", "teen nick", "teennick",
  "chat show", "film detective", "tom and jerry", "tom jerry", "cipher", "ion mystery",
  "wfxt", "whiplash", "amc absolute",
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
    const hits = channels.filter((c) => c.name.toLowerCase().includes(term) || c.id.toLowerCase().includes(term.replace(/\s/g, "")));
    if (hits.length) {
      console.log(`  [${term}]`);
      for (const h of hits.slice(0, 5)) console.log(`    ${h.name} | ${h.id}`);
    }
  }
}
