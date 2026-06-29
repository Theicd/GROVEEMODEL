import { gunzipSync } from "node:zlib";

function parseDate(raw) {
  const mm = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!mm) return null;
  const [, y, mo, d, h, mi, s, tz] = mm;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  return new Date(iso);
}

function findChannelId(xml, nameRe) {
  const chRe = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/gi;
  let m;
  while ((m = chRe.exec(xml))) {
    if (nameRe.test(m[2])) return { id: m[1], name: m[2] };
  }
  return null;
}

function liveProgram(xml, chId) {
  const esc = chId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`<programme\\s([^>]*?)channel="${esc}"([^>]*?)>([\\s\\S]*?)</programme>`, "gi");
  const now = Date.now();
  let m;
  let live = null;
  while ((m = re.exec(xml))) {
    const attrs = m[1] + m[2];
    const st = parseDate(attrs.match(/start="([^"]+)"/i)?.[1] || "");
    const en = parseDate(attrs.match(/stop="([^"]+)"/i)?.[1] || "");
    if (!st || !en) continue;
    if (st.getTime() <= now && en.getTime() > now) {
      const title = m[3].match(/<title>([^<]*)</i)?.[1];
      live = { title, st, en, mins: Math.round((en - st) / 60000) };
    }
  }
  return live;
}

const sources = [
  ["Plex", "https://i.mjh.nz/Plex/us.xml.gz"],
  ["Samsung", "https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],
];

for (const [label, url] of sources) {
  const xml = gunzipSync(Buffer.from(await (await fetch(url)).arrayBuffer())).toString("utf8");
  const ch = findChannelId(xml, /^Gravitas Movies$/i);
  if (!ch) {
    console.log(label, "no channel");
    continue;
  }
  const live = liveProgram(xml, ch.id);
  console.log(`\n${label} (${ch.id}):`);
  if (!live) console.log("  nothing live");
  else {
    console.log("  TITLE:", live.title);
    console.log("  UTC:", live.st.toISOString(), "->", live.en.toISOString());
    console.log("  IL:", live.st.toLocaleString("he-IL", { timeZone: "Asia/Jerusalem" }), "->", live.en.toLocaleString("he-IL", { timeZone: "Asia/Jerusalem" }));
    console.log("  slot minutes:", live.mins);
  }
}

// media playlist PDT
const master = await (await fetch("https://d6dg3ebeih71x.cloudfront.net/Gravitas_Movies.m3u8")).text();
const variant = master.match(/^Gravitas_Movies1080p\.m3u8/m)?.[0];
if (variant) {
  const pl = await (await fetch(`https://d6dg3ebeih71x.cloudfront.net/${variant}`)).text();
  const pdt = pl.match(/#EXT-X-PROGRAM-DATE-TIME:(.+)/)?.[1];
  console.log("\nHLS PROGRAM-DATE-TIME:", pdt ?? "none");
  console.log(pl.split("\n").slice(0, 15).join("\n"));
}
