import { gunzipSync } from "node:zlib";

const SOURCES = [
  ["roku", "https://i.mjh.nz/Roku/all.xml.gz"],
  ["pluto", "https://i.mjh.nz/PlutoTV/us.xml.gz"],
  ["plex", "https://i.mjh.nz/Plex/us.xml.gz"],
  ["samsung", "https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],
];

for (const [key, url] of SOURCES) {
  const buf = Buffer.from(await (await fetch(url)).arrayBuffer());
  const xml = gunzipSync(buf).toString("utf8");
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  console.log(`\n=== ${key} ===`);
  while ((m = re.exec(xml))) {
    const n = m[2].toLowerCase();
    if (n.includes("entertain") || n === "et" || n.includes("tonight")) {
      console.log(m[2], "|", m[1]);
    }
  }
}
