import { gunzipSync } from "node:zlib";

const buf = Buffer.from(await (await fetch("https://i.mjh.nz/Roku/all.xml.gz")).arrayBuffer());
const xml = gunzipSync(buf).toString("utf8");
const id = "16f751e2330d5a09a5e1a25a52b2b09c";

function parse(raw, offsetHours = 0) {
  const m = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s, tz] = m;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  const dt = new Date(iso);
  if (offsetHours) dt.setTime(dt.getTime() + offsetHours * 3600000);
  return dt;
}

const re = new RegExp(`<programme\\s([^>]*?)>([\\s\\S]*?)</programme>`, "gi");
const now = new Date();
console.log("now", now.toISOString());

for (const offset of [0, 4, -4, 5]) {
  let live = null;
  let m;
  re.lastIndex = 0;
  while ((m = re.exec(xml))) {
    if (!m[1].includes(id)) continue;
    const start = parse(m[1].match(/start="([^"]+)"/i)?.[1] ?? "", offset);
    const stop = parse(m[1].match(/stop="([^"]+)"/i)?.[1] ?? "", offset);
    const title = m[2].match(/<title>([^<]*)</i)?.[1];
    if (start && stop && start <= now && stop > now) live = { title, start, stop };
  }
  console.log(`offset ${offset}h:`, live?.title, live?.start?.toISOString());
}

// slots around now without offset
re.lastIndex = 0;
const slots = [];
while ((m = re.exec(xml))) {
  if (!m[1].includes(id)) continue;
  const start = parse(m[1].match(/start="([^"]+)"/i)?.[1] ?? "");
  const title = m[2].match(/<title>([^<]*)</i)?.[1];
  if (start && Math.abs(start.getTime() - now.getTime()) < 3 * 3600000) slots.push({ title, start: start.toISOString() });
}
console.log("\nSlots within 3h (raw UTC parse):");
for (const s of slots) console.log(s.start, s.title);
