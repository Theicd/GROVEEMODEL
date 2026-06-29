import { gunzipSync } from "node:zlib";

const url = "https://i.mjh.nz/SamsungTVPlus/us.xml.gz";
const buf = Buffer.from(await (await fetch(url)).arrayBuffer());
const xml = gunzipSync(buf).toString("utf8");

const chRe = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/gi;
let m;
let chId = null;
while ((m = chRe.exec(xml))) {
  if (/gravitas/i.test(m[2])) {
    chId = m[1];
    console.log("CH", m[1], m[2]);
    break;
  }
}
if (!chId) {
  console.log("no channel");
  process.exit(1);
}

function parseDate(raw) {
  const mm = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!mm) return null;
  const [, y, mo, d, h, mi, s, tz] = mm;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  return new Date(iso);
}

const esc = chId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const re = new RegExp(`<programme\\s([^>]*?)channel="${esc}"([^>]*?)>([\\s\\S]*?)</programme>`, "gi");
const now = Date.now();
let n = 0;
while ((m = re.exec(xml))) {
  const attrs = m[1] + m[2];
  const st = parseDate(attrs.match(/start="([^"]+)"/i)?.[1] || "");
  const en = parseDate(attrs.match(/stop="([^"]+)"/i)?.[1] || "");
  if (!st || !en) continue;
  if (en.getTime() < now - 3600000) continue;
  if (st.getTime() > now + 7200000) continue;
  const block = m[3];
  const title = block.match(/<title>([^<]*)</i)?.[1];
  const lenTag = block.match(/<length[^>]*>([^<]*)</i)?.[1];
  const lenUnits = block.match(/<length[^>]*units="([^"]+)"/i)?.[1];
  const mins = Math.round((en - st) / 60000);
  const live = st.getTime() <= now && en.getTime() > now;
  console.log(
    live ? ">>> LIVE" : "    ",
    "IL start",
    st.toLocaleString("he-IL", { timeZone: "Asia/Jerusalem" }),
    "IL end",
    en.toLocaleString("he-IL", { timeZone: "Asia/Jerusalem" }),
    `slot=${mins}min`,
    lenTag ? `length=${lenTag}${lenUnits ? ` ${lenUnits}` : ""}` : "",
    title,
  );
  if (++n >= 12) break;
}
