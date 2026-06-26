/**
 * Probe every curated favorite against MJH EPG — channel by channel.
 * Run: node scripts/probe-all-favorites.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dir = dirname(fileURLToPath(import.meta.url));
const root = join(__dir, "..");
const favorites = JSON.parse(readFileSync(join(root, "public/liveMedia/curatedFavorites.json"), "utf8"));

const SOURCES = [
  { key: "mjh-all", label: "MJH All", url: "https://i.mjh.nz/all/epg.xml.gz" },
  { key: "mjh-plex-us", label: "Plex US", url: "https://i.mjh.nz/Plex/us.xml.gz", hint: /plex\.|wurl\.tv|amagi\.tv|mediatailor/i },
  { key: "mjh-pluto-us", label: "Pluto US", url: "https://i.mjh.nz/PlutoTV/us.xml.gz", hint: /pluto\.tv/i },
  { key: "mjh-samsung-us", label: "Samsung US", url: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz", hint: /samsung/i },
  { key: "mjh-roku", label: "Roku", url: "https://i.mjh.nz/Roku/all.xml.gz", hint: /roku/i },
];

const ALIASES = {
  cops: "Cops.us",
  "the pet collective": "ThePetCollective.us",
  "pet collective": "ThePetCollective.us",
  failarmy: "FailArmy.us",
  "red bull tv": "RedBullTV.us",
  "comedy central": "ComedyCentral.us",
  "bbc top gear": "BBCTopGear.uk",
  moviesphere: "MovieSphere.us",
  "movie sphere": "MovieSphere.us",
  "now 14": "Now14.il",
  "reshet 13": "Channel13.il",
  "channel 13": "Channel13.il",
  "ion mystery": "IonMystery.us",
  "wfxt-dt2": "IonMystery.us",
  "abc kids": "ABCKids.au",
  "fifa+ united states": "FIFAPlus.uk",
  "fifa+": "FIFAPlus.uk",
  charge: "Charge.us",
  "entertainment tonight": "EntertainmentTonight.us",
  "amc absolute reality": "AbsoluteRealitybyWETV.us",
  "global fashion channel": "GlobalFashionChannel.us",
  "space series": "SpaceSeries.us",
};
const HINTS = {
  "Cops.us": ["cops", "cops tv"],
  "ThePetCollective.us": ["the pet collective", "pet collective"],
  "BBCTopGear.uk": ["bbc top gear", "top gear"],
  "RedBullTV.us": ["red bull tv", "redbull tv"],
  "MovieSphere.us": ["moviesphere", "movie sphere"],
  "IonMystery.us": ["ion mystery", "on mystery", "wfxt-dt2", "wfxt dt2"],
  "WFXT662.us": ["ion mystery", "on mystery", "wfxt-dt2", "wfxt dt2"],
  "ABCKids.au": ["abc kids"],
  "FIFAPlus.uk": ["FIFA+", "fifa+", "fifa plus"],
  "Charge.us": ["charge!", "charge"],
  "EntertainmentTonight.us": ["Entertainment Tonight", "entertainment tonight", "ET"],
  "AbsoluteRealitybyWETV.us": ["all reality we tv", "absolute reality", "amc absolute reality"],
  "GlobalFashionChannel.us": ["global fashion", "fashiontv"],
  "SpaceSeries.us": ["space series", "space & beyond"],
};

function normalizeTitle(raw) {
  return raw
    .replace(/like Gecko\)[^,]*,\s*/gi, "")
    .replace(/\s*\(\d+p\)\s*/gi, " ")
    .replace(/\s*\[[^\]]*\]\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function normalizeForMatch(raw) {
  return normalizeTitle(raw).toLowerCase().replace(/[^a-z0-9\u0590-\u05ff]+/g, " ").replace(/\s+/g, " ").trim();
}
function resolveOrgId(title, tvgId) {
  if (tvgId?.trim()) {
    const base = tvgId.includes("@") ? tvgId.split("@")[0] : tvgId;
    if (base.trim()) return base.trim();
  }
  const key = normalizeForMatch(title);
  if (ALIASES[key]) return ALIASES[key];
  for (const [alias, id] of Object.entries(ALIASES)) {
    if (key.includes(alias) || alias.includes(key)) return ALIASES[alias];
  }
  return null;
}
function resolveEpgMatchTitles(title, tvgId, streamUrl) {
  const out = [];
  const add = (v) => {
    const n = normalizeTitle(v);
    if (n && !out.includes(n)) out.push(n);
  };
  add(title);
  const norm = normalizeForMatch(title);
  const orgId = resolveOrgId(title, tvgId);
  if (orgId) for (const hint of HINTS[orgId] ?? []) add(hint);
  if (tvgId?.trim()) {
    const tvgBase = tvgId.includes("@") ? tvgId.split("@")[0].trim() : tvgId.trim();
    for (const hint of HINTS[tvgBase] ?? []) add(hint);
  }
  for (const [alias, id] of Object.entries(ALIASES)) {
    if (norm === alias || norm.includes(alias) || alias.includes(norm)) {
      for (const hint of HINTS[id] ?? []) add(hint);
    }
  }
  if (/ionmystery/i.test(streamUrl ?? "")) add("ION Mystery");
  if (/abc-kids|abckids/i.test(streamUrl ?? "")) add("ABC Kids");
  if (/sysdata_s_p_a_fifa|fifa[_-]?plus/i.test(streamUrl ?? "")) add("FIFA+");
  if (/absolutereality|wetv/i.test(streamUrl ?? "")) add("All Reality We TV");
  if (/enterbcef|entertainmenttonight/i.test(streamUrl ?? "")) add("ET");
  if (/globalfashionchannel|pubgfc/i.test(streamUrl ?? "")) add("FashionTV");
  if (/sofast\.tv/i.test(streamUrl ?? "")) add("Space & Beyond");
  return out;
}
function resolveKeys(title) {
  const keys = [];
  const add = (v) => {
    if (!v?.trim()) return;
    const n = normalizeForMatch(v);
    if (n && !keys.includes(n)) keys.push(n);
    const l = v.toLowerCase();
    if (!keys.includes(l)) keys.push(l);
  };
  add(normalizeForMatch(title));
  add(title);
  const org = resolveOrgId(title);
  add(org);
  if (org) for (const h of HINTS[org] ?? []) add(h);
  return keys;
}

function tokensOf(s) {
  return s.toLowerCase().split(/[^a-z0-9\u0590-\u05ff]+/).filter(Boolean);
}

function parseChannels(xml, sourceKey) {
  const out = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  while ((m = re.exec(xml))) out.push({ id: m[1], name: m[2], sourceKey });
  return out;
}

function stripTvgFeed(tvgId) {
  const at = tvgId.indexOf("@");
  return at >= 0 ? tvgId.slice(0, at) : tvgId;
}

const GENERIC_TOKENS = new Set([
  "movies", "movie", "tv", "television", "channel", "live", "news", "sports", "sport", "music", "plus",
  "the", "and", "hd", "free", "international", "network", "entertainment", "family", "kids", "classic",
  "world", "america", "american", "usa", "uk", "video", "on", "demand",
]);
const STREAM_PATH_IGNORE = new Set([
  "master", "playlist", "index", "live", "stream", "hls", "chunklist", "manifest", "media", "pri", "sd", "hd", "m3u8",
]);
const MIN_SCORE = 65;

function significantTokens(s) {
  return tokensOf(s).filter((t) => !GENERIC_TOKENS.has(t) && t.length >= 2);
}
function nameHasToken(channelName, token) {
  return significantTokens(channelName).includes(token);
}
function compactAlpha(s) {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}
function tvgIdVariants(tvgId) {
  if (!tvgId?.trim()) return [];
  const bare = stripTvgFeed(tvgId.trim());
  const out = [compactAlpha(bare)];
  const noRegion = bare.replace(/\.[a-z]{2}$/i, "");
  const compact = compactAlpha(noRegion);
  if (compact && !out.includes(compact)) out.push(compact);
  return out.filter((v) => v.length >= 4);
}
function streamPathTokens(streamUrl) {
  if (!streamUrl) return [];
  try {
    const base = new URL(streamUrl).pathname.split("/").pop()?.replace(/\.m3u8$/i, "") ?? "";
    return significantTokens(base.replace(/_/g, " ")).filter((t) => !STREAM_PATH_IGNORE.has(t));
  } catch {
    return [];
  }
}
function scoreChannelMatch(channel, title, tvgId, streamUrl) {
  const normTitle = normalizeForMatch(title);
  const normName = normalizeForMatch(channel.name);
  if (normTitle && normTitle === normName) return 100;
  const idCompact = compactAlpha(channel.id);
  const nameCompact = compactAlpha(channel.name);
  for (const variant of tvgIdVariants(tvgId)) {
    if (idCompact.includes(variant) || nameCompact.includes(variant)) return 96;
  }
  const titleSig = significantTokens(title);
  const nameSig = significantTokens(channel.name);
  const pathSig = streamPathTokens(streamUrl);
  if (titleSig.length >= 2 && titleSig.every((t) => nameHasToken(channel.name, t))) return 85 + titleSig.length * 4;
  if (titleSig.length === 1 && titleSig[0].length >= 6 && nameHasToken(channel.name, titleSig[0])) return 84;
  if (nameSig.length >= 2 && nameSig.every((t) => tokensOf(title).includes(t))) return 80 + nameSig.length * 4;
  if (pathSig.length >= 2 && pathSig.every((t) => nameHasToken(channel.name, t))) return 78 + pathSig.length * 4;
  if (pathSig.length === 1 && pathSig[0].length >= 6 && nameHasToken(channel.name, pathSig[0])) return 77;
  const sharedSig = titleSig.filter((t) => nameSig.includes(t));
  if (sharedSig.length >= 2) return 70 + sharedSig.length * 5;
  return 0;
}
function findBestChannelMatch(channels, title, tvgId, streamUrl) {
  let best = null;
  for (const channel of channels) {
    const score = scoreChannelMatch(channel, title, tvgId, streamUrl);
    if (score < MIN_SCORE) continue;
    if (!best || score > best.score) best = { channel, score };
  }
  return best?.channel ?? null;
}

function matchChannel(channels, title, tvgId, streamUrl) {
  return findBestChannelMatch(channels, title, tvgId, streamUrl);
}

function parseXmltvDate(raw) {
  const m = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s, tz] = m;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  const dt = new Date(iso);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

function countPrograms(xml, channelId, upcomingOnly = false) {
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`<programme\\s[^>]*channel="${escaped}"[^>]*start="([^"]+)"[^>]*stop="([^"]+)"`, "g");
  const now = Date.now();
  const horizon = now - 6 * 60 * 60 * 1000;
  let total = 0;
  let upcoming = 0;
  let firstTitle = null;
  let m;
  const titleRe = new RegExp(`<programme\\s[^>]*channel="${escaped}"[^>]*>[\\s\\S]*?<title>([^<]*)</title>`, "i");
  const titleM = titleRe.exec(xml);
  if (titleM) firstTitle = titleM[1];

  while ((m = re.exec(xml))) {
    total++;
    const end = parseXmltvDate(m[2]);
    if (end && end.getTime() > horizon) upcoming++;
  }
  return { total, upcoming, firstTitle };
}

function orderedSources(stream) {
  const all = SOURCES[0];
  const hinted = SOURCES.slice(1).filter((s) => s.hint?.test(stream));
  const rest = SOURCES.slice(1).filter((s) => !s.hint?.test(stream));
  return [all, ...hinted, ...rest];
}

console.log("Loading MJH XMLTV sources...");
const xmlByKey = new Map();
for (const src of SOURCES) {
  process.stdout.write(`  ${src.label}... `);
  const t0 = Date.now();
  const res = await fetch(src.url);
  const buf = Buffer.from(await res.arrayBuffer());
  const xml = src.url.endsWith(".gz") ? gunzipSync(buf).toString("utf8") : buf.toString("utf8");
  xmlByKey.set(src.key, xml);
  const ch = parseChannels(xml, src.key).length;
  console.log(`${ch} channels (${Date.now() - t0}ms)`);
}

const channelsByKey = new Map();
for (const src of SOURCES) {
  channelsByKey.set(src.key, parseChannels(xmlByKey.get(src.key), src.key));
}

const results = [];
for (const ch of favorites.channels) {
  const title = normalizeTitle(ch.name);
  const orgId = resolveOrgId(title, ch.tvgId);
  if (orgId && /\.il$/i.test(orgId)) {
    results.push({ name: ch.name, status: "NO_FEED", orgId, reason: "Israeli — no open MJH feed" });
    continue;
  }
  const matchTitles = resolveEpgMatchTitles(title, ch.tvgId, ch.stream);
  const sources = orderedSources(ch.stream);
  let found = null;
  let stats = null;
  let sourceLabel = null;
  let bestScore = 0;

  for (const src of sources) {
    const xml = xmlByKey.get(src.key);
    const channels = channelsByKey.get(src.key);
    for (const matchTitle of matchTitles) {
      const matched = matchChannel(channels, matchTitle, ch.tvgId, ch.stream);
      if (!matched) continue;
      const score = scoreChannelMatch(matched, matchTitle, ch.tvgId, ch.stream);
      if (score < bestScore) continue;
      const s = countPrograms(xml, matched.id);
      if (s.total > 0 && score >= bestScore) {
        found = matched;
        stats = s;
        sourceLabel = src.label;
        bestScore = score;
      }
    }
  }

  if (!found) {
    results.push({ name: ch.name, status: "NO_MATCH", orgId, keys: matchTitles.slice(0, 4) });
  } else if (stats.upcoming === 0) {
    results.push({
      name: ch.name,
      status: "MATCH_NO_UPCOMING",
      orgId,
      epgName: found.name,
      epgId: found.id,
      source: sourceLabel,
      totalProgs: stats.total,
      sample: stats.firstTitle,
    });
  } else {
    results.push({
      name: ch.name,
      status: "OK",
      orgId,
      epgName: found.name,
      epgId: found.id,
      source: sourceLabel,
      totalProgs: stats.total,
      upcoming: stats.upcoming,
      sample: stats.firstTitle,
    });
  }
}

const ok = results.filter((r) => r.status === "OK");
const noUp = results.filter((r) => r.status === "MATCH_NO_UPCOMING");
const noMatch = results.filter((r) => r.status === "NO_MATCH");
const noFeed = results.filter((r) => r.status === "NO_FEED");

console.log("\n=== SUMMARY ===");
console.log(`Total favorites: ${results.length}`);
console.log(`OK (has upcoming programmes): ${ok.length}`);
console.log(`Matched but no upcoming: ${noUp.length}`);
console.log(`No EPG match: ${noMatch.length}`);
console.log(`No feed (IL): ${noFeed.length}`);

console.log("\n=== OK ===");
for (const r of ok) console.log(`  ✓ ${r.name} → ${r.epgName} [${r.source}] (${r.upcoming} upcoming, sample: ${r.sample})`);

console.log("\n=== MATCH BUT NO UPCOMING (UI shows empty!) ===");
for (const r of noUp) console.log(`  ⚠ ${r.name} → ${r.epgName} [${r.source}] (${r.totalProgs} total)`);

console.log("\n=== NO MATCH ===");
for (const r of noMatch) console.log(`  ✗ ${r.name}`);

writeFileSync(join(root, "scripts/probe-all-favorites-results.json"), JSON.stringify(results, null, 2));
console.log("\nFull results: scripts/probe-all-favorites-results.json");
