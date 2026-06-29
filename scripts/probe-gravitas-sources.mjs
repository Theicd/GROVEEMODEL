import { gunzipSync } from "node:zlib";

const SOURCES = [
  ["mjh-all", "https://i.mjh.nz/all/epg.xml.gz"],
  ["plex", "https://i.mjh.nz/Plex/us.xml.gz"],
  ["pluto", "https://i.mjh.nz/PlutoTV/us.xml.gz"],
  ["samsung", "https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],
  ["roku", "https://i.mjh.nz/Roku/all.xml.gz"],
];

for (const [name, url] of SOURCES) {
  const xml = gunzipSync(Buffer.from(await (await fetch(url)).arrayBuffer())).toString("utf8");
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/gi;
  let m;
  const hits = [];
  while ((m = re.exec(xml))) {
    if (/gravitas/i.test(m[2]) || /gravitas/i.test(m[1])) hits.push({ id: m[1], name: m[2] });
  }
  if (hits.length) {
    console.log("\n===", name, "===");
    for (const h of hits) console.log(h.id, h.name);
  }
}
