import { gunzipSync } from "node:zlib";

const URLS = {
  all: "https://i.mjh.nz/all/epg.xml.gz",
  plex: "https://i.mjh.nz/Plex/us.xml.gz",
  roku: "https://i.mjh.nz/Roku/all.xml.gz",
  samsung: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz",
};

async function fetchXml(key) {
  const buf = Buffer.from(await (await fetch(URLS[key])).arrayBuffer());
  return gunzipSync(buf).toString("utf8");
}

function findChannels(xml, term) {
  const re = /<channel id="([^"]+)"[^>]*>([\s\S]*?)<\/channel>/g;
  const hits = [];
  let m;
  while ((m = re.exec(xml))) {
    const block = m[0];
    if (!block.toLowerCase().includes(term.toLowerCase())) continue;
    const names = [...block.matchAll(/<display-name>([^<]*)<\/display-name>/g)].map((x) => x[1]);
    hits.push({ id: m[1], names });
  }
  return hits;
}

function parseXmltvDate(raw) {
  const p = raw.match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/);
  if (!p) return null;
  return new Date(`${p[1]}-${p[2]}-${p[3]}T${p[4]}:${p[5]}:${p[6]}Z`);
}

function nowProgrammes(xml, channelId, limit = 8) {
  const now = Date.now();
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(
    `<programme\\s[^>]*channel="${escaped}"[^>]*start="([^"]+)"[^>]*stop="([^"]+)"[^>]*>([\\s\\S]*?)<\\/programme>`,
    "g",
  );
  const progs = [];
  let m;
  while ((m = re.exec(xml))) {
    const [, start, stop, body] = m;
    const title = (body.match(/<title[^>]*>([^<]*)<\/title>/) || [])[1] || "";
    const desc = (body.match(/<desc[^>]*>([^<]*)<\/desc>/) || [])[1] || "";
    const s = parseXmltvDate(start);
    const e = parseXmltvDate(stop);
    if (!s || !e) continue;
    if (e.getTime() < now - 3600000) continue;
    if (s.getTime() > now + 86400000) continue;
    progs.push({ start, stop, title, desc: desc.slice(0, 160), s: s.getTime() });
  }
  progs.sort((a, b) => a.s - b.s);
  return progs.slice(0, limit);
}

const RAKUTEN_UK =
  "https://raw.githubusercontent.com/dp247/rakuten-uk-epg/master/epg.xml";

console.log("\n=== rakuten-uk (dp247) ===");
{
  const xml = await (await fetch(RAKUTEN_UK)).text();
  for (const term of ["history hunters", "historyhunters"]) {
    const hits = findChannels(xml, term);
    console.log(`term "${term}":`, hits.length, "hits");
    for (const h of hits) {
      console.log("  channel:", h.id, "|", h.names.join(" / "));
      const progs = nowProgrammes(xml, h.id, 10);
      for (const p of progs) console.log("   ", p.start, "-", p.stop, "|", p.title, "|", p.desc.slice(0, 100));
    }
  }
}

for (const key of Object.keys(URLS)) {
  console.log(`\n=== ${key} ===`);
  const xml = await fetchXml(key);
  for (const term of ["history hunt", "historyhunters", "rakuten"]) {
    const hits = findChannels(xml, term);
    if (!hits.length) continue;
    console.log(`term "${term}":`, hits.length, "hits");
    for (const h of hits.slice(0, 5)) {
      console.log("  channel:", h.id, "|", h.names.join(" / "));
      const progs = nowProgrammes(xml, h.id);
      for (const p of progs) console.log("   ", p.start, "-", p.stop, "|", p.title, "|", p.desc.slice(0, 80));
    }
  }
}
