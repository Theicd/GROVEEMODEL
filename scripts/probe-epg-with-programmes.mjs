import { gunzipSync } from "node:zlib";

const SEARCHES = [
  "entertainment tonight", "charge", "discoverfilm", "discover film", "film detective",
  "abc kids", "fifa", "positiv", "rally", "deluxe music", "inwild", "whiplash",
  "history hunters", "chat show", "tom and jerry", "cipher", "groovy", "classique",
  "mix hollywood", "tunebox", "fnx", "kvcr", "30a", "autentic", "all reality",
  "absolute reality", "teen nick", "teennick",
];

const SOURCES = [
  ["plex", "https://i.mjh.nz/Plex/us.xml.gz"],
  ["pluto", "https://i.mjh.nz/PlutoTV/us.xml.gz"],
  ["samsung", "https://i.mjh.nz/SamsungTVPlus/us.xml.gz"],
  ["roku", "https://i.mjh.nz/Roku/all.xml.gz"],
  ["all", "https://i.mjh.nz/all/epg.xml.gz"],
];

function parseXmltvDate(raw) {
  const m = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s, tz] = m;
  return new Date(`${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`);
}

function countProgs(xml, channelId) {
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`<programme\\s[^>]*channel="${escaped}"[^>]*stop="([^"]+)"`, "g");
  const horizon = Date.now() - 6 * 60 * 60 * 1000;
  let total = 0;
  let upcoming = 0;
  let m;
  while ((m = re.exec(xml))) {
    total++;
    const end = parseXmltvDate(m[1]);
    if (end && end.getTime() > horizon) upcoming++;
  }
  return { total, upcoming };
}

for (const [key, url] of SOURCES) {
  const buf = Buffer.from(await (await fetch(url)).arrayBuffer());
  const xml = gunzipSync(buf).toString("utf8");
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  const channels = [];
  let m;
  while ((m = re.exec(xml))) channels.push({ id: m[1], name: m[2] });

  console.log(`\n=== ${key} ===`);
  for (const term of SEARCHES) {
    const hits = channels.filter((c) => c.name.toLowerCase().includes(term));
    for (const h of hits.slice(0, 3)) {
      const stats = countProgs(xml, h.id);
      if (stats.upcoming > 0) {
        console.log(`  [${term}] ${h.name} | upcoming=${stats.upcoming} | ${h.id.slice(0, 55)}`);
      }
    }
  }
}
