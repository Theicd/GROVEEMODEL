import { gunzipSync } from "node:zlib";

const IDS = [
  ["roku-et", "roku", "392a421311b35de594961680be62564c"],
  ["pluto-et", "pluto", "5dc0c78281eddb0009a02d5e"],
  ["samsung-et", "samsung", "USBA3700002JF"],
];

const URLS = {
  roku: "https://i.mjh.nz/Roku/all.xml.gz",
  pluto: "https://i.mjh.nz/PlutoTV/us.xml.gz",
  samsung: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz",
};

for (const [label, src, id] of IDS) {
  const buf = Buffer.from(await (await fetch(URLS[src])).arrayBuffer());
  const xml = gunzipSync(buf).toString("utf8");
  const re = new RegExp(`<programme\\s[^>]*channel="${id}"[^>]*>[\\s\\S]*?<title>([^<]*)</title>`, "i");
  const m = re.exec(xml);
  console.log(label, m ? m[1] : "no programmes");
}
