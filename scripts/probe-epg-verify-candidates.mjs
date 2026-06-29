import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

const CANDIDATES = [
  { name: "ABC Kids", src: "all", id: "mjh-abc-kids" },
  { name: "AMC Absolute Reality", src: "plex", id: "6a1610bebdf296985fd95603-need-find" },
  { name: "Entertainment Tonight", src: "pluto", id: "need-find-et" },
  { name: "FIFA+", src: "all", id: "mjh-sbs-sysdata_s_p_a_fifa_6" },
  { name: "Global Fashion→FashionTV", src: "plex", search: "FashionTV" },
  { name: "Space→Space & Beyond", src: "plex", search: "Space &amp; Beyond" },
  { name: "WFXT→Ion Mystery", src: "plex", search: "Ion Mystery" },
  { name: "Charge", src: "all", search: "charge" },
  { name: "DiscoverFilm", src: "samsung", search: "discover" },
  { name: "Positiv", src: "all", search: "positiv" },
  { name: "Rally", src: "all", search: "rally" },
  { name: "Film Detective", src: "plex", search: "Film Detective" },
  { name: "Tom Jerry", src: "all", search: "tom" },
  { name: "Deluxe Music", src: "all", search: "deluxe" },
  { name: "INWILD", src: "all", search: "inwild" },
  { name: "Whiplash", src: "all", search: "whiplash" },
  { name: "History Hunters", src: "all", search: "history hunt" },
  { name: "Chat Show", src: "all", search: "chat show" },
  { name: "30A Classic", src: "all", search: "30a" },
  { name: "Autentic", src: "all", search: "autentic" },
  { name: "Cipher", src: "all", search: "cipher" },
  { name: "Groovy", src: "all", search: "groovy" },
  { name: "Classique", src: "all", search: "classique" },
  { name: "Mix Hollywood", src: "all", search: "mix hollywood" },
  { name: "TuneBox", src: "all", search: "tunebox" },
  { name: "FNX/KVCR", src: "all", search: "fnx" },
];

const URLS = {
  all: "https://i.mjh.nz/all/epg.xml.gz",
  plex: "https://i.mjh.nz/Plex/us.xml.gz",
  pluto: "https://i.mjh.nz/PlutoTV/us.xml.gz",
  samsung: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz",
  roku: "https://i.mjh.nz/Roku/all.xml.gz",
};

function parseXmltvDate(raw) {
  const m = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s, tz] = m;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  const dt = new Date(iso);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

function countProgs(xml, channelId) {
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`<programme\\s[^>]*channel="${escaped}"[^>]*stop="([^"]+)"`, "g");
  const now = Date.now();
  const horizon = now - 6 * 60 * 60 * 1000;
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

function findChannel(xml, search) {
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  const hits = [];
  while ((m = re.exec(xml))) {
    if (m[2].toLowerCase().includes(search.toLowerCase()) || m[1].toLowerCase().includes(search.toLowerCase())) {
      hits.push({ id: m[1], name: m[2] });
    }
  }
  return hits;
}

const xmlCache = {};
for (const key of new Set(CANDIDATES.map((c) => c.src))) {
  const buf = Buffer.from(await (await fetch(URLS[key])).arrayBuffer());
  xmlCache[key] = gunzipSync(buf).toString("utf8");
}

for (const c of CANDIDATES) {
  const xml = xmlCache[c.src];
  let channels = [];
  if (c.id && !c.id.includes("need")) {
    channels = [{ id: c.id, name: c.id }];
  } else if (c.search) {
    channels = findChannel(xml, c.search).slice(0, 3);
  }
  if (!channels.length) {
    console.log(`${c.name}: NOT FOUND in ${c.src}`);
    continue;
  }
  for (const ch of channels) {
    const stats = countProgs(xml, ch.id);
    console.log(`${c.name} | ${ch.name} | ${ch.id.slice(0, 50)} | total=${stats.total} upcoming=${stats.upcoming}`);
  }
}
