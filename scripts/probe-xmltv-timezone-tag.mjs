import { gunzipSync } from "node:zlib";

const buf = Buffer.from(await (await fetch("https://i.mjh.nz/Roku/all.xml.gz")).arrayBuffer());
const xml = gunzipSync(buf).toString("utf8");

const hasTz = xml.includes("<timezone");
console.log("has timezone tag:", hasTz);
if (hasTz) {
  const m = xml.match(/<timezone>([^<]*)<\/timezone>/);
  console.log("sample:", m?.[1]);
}

const ch = xml.match(/<channel id="16f751e2330d5a09a5e1a25a52b2b09c"[^>]*>[\s\S]*?<\/channel>/);
console.log("ion channel block has tz:", ch?.[0].includes("timezone"));
