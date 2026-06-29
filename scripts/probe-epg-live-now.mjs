import { gunzipSync } from "node:zlib";

const buf = Buffer.from(await (await fetch("https://i.mjh.nz/Roku/all.xml.gz")).arrayBuffer());
const xml = gunzipSync(buf).toString("utf8");

const idx = xml.indexOf("16f751e2330d5a09a5e1a25a52b2b09c");
console.log("sample raw:", xml.slice(idx - 80, idx + 400));

const id = "16f751e2330d5a09a5e1a25a52b2b09c";
const re = new RegExp(`<programme\\s([^>]*?)>([\\s\\S]*?)</programme>`, "gi");
let m;
let c = 0;
while ((m = re.exec(xml)) && c < 5) {
  if (!m[1].includes(id)) continue;
  const title = m[2].match(/<title>([^<]*)</i)?.[1];
  const start = m[1].match(/start="([^"]+)"/i)?.[1];
  const stop = m[1].match(/stop="([^"]+)"/i)?.[1];
  console.log(start, stop, title);
  c++;
}

function parseXmltvDate(raw) {
  const t = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!t) return null;
  const [, y, mo, d, h, mi, s, tz] = t;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  return new Date(iso);
}

const now = new Date();
console.log("\nNow UTC:", now.toISOString());
console.log("Now IL:", now.toLocaleString("he-IL", { timeZone: "Asia/Jerusalem" }));

// find live on ion mystery
c = 0;
re.lastIndex = 0;
while ((m = re.exec(xml))) {
  if (!m[1].includes(id)) continue;
  const start = parseXmltvDate(m[1].match(/start="([^"]+)"/i)?.[1] ?? "");
  const stop = parseXmltvDate(m[1].match(/stop="([^"]+)"/i)?.[1] ?? "");
  const title = m[2].match(/<title>([^<]*)</i)?.[1];
  if (start && stop && start <= now && stop > now) {
    console.log("LIVE EPG:", title, start.toISOString(), "->", stop.toISOString());
    c++;
  }
}
