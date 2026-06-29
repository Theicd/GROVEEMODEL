import { gunzipSync } from "node:zlib";

const res = await fetch("https://i.mjh.nz/all/epg.xml.gz");
const xml = gunzipSync(Buffer.from(await res.arrayBuffer())).toString("utf8");
const now = new Date();
const id = "mjh-10-cops";
let live = 0;
let future = 0;
const re = /<programme channel="mjh-10-cops" start="([^"]+)" stop="([^"]+)"/g;
let m;
while ((m = re.exec(xml))) {
  const parse = (raw) => {
    const t = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/);
    if (!t) return null;
    return new Date(Date.UTC(+t[1], +t[2] - 1, +t[3], +t[4], +t[5], +t[6]));
  };
  const st = parse(m[1]);
  const en = parse(m[2]);
  if (!st || !en) continue;
  if (st <= now && en > now) live++;
  if (st > now) future++;
}
console.log("now", now.toISOString(), { live, future });
