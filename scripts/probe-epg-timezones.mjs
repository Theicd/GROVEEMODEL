import { gunzipSync } from "node:zlib";

const buf = Buffer.from(await (await fetch("https://i.mjh.nz/Roku/all.xml.gz")).arrayBuffer());
const xml = gunzipSync(buf).toString("utf8");

// ION Mystery + COPS samples
const ids = ["16f751e2330d5a09a5e1a25a52b2b09c", "7f28cd98ab575aa784d0f80698279e70"];

for (const id of ids) {
  console.log(`\n=== ${id} ===`);
  const re = new RegExp(`<programme start="([^"]+)" stop="([^"]+)" channel="${id}"[^>]*>[\\s\\S]*?<title>([^<]*)</title>`, "g");
  let m;
  let c = 0;
  while ((m = re.exec(xml)) && c < 6) {
    console.log(m[1], "->", m[2], "|", m[3]);
    c++;
  }
}

// timezone suffix distribution
const tzRe = /start="(\d{14})\s*([+-]\d{4})?"/g;
const tzCounts = new Map();
let m;
while ((m = tzRe.exec(xml))) {
  const tz = m[2] ?? "(none)";
  tzCounts.set(tz, (tzCounts.get(tz) ?? 0) + 1);
}
console.log("\n=== TZ distribution ===");
for (const [tz, n] of [...tzCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)) {
  console.log(tz, n);
}
