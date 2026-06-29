/**
 * Probe favorites using same match-title logic as epgService (resolveEpgMatchTitles).
 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

// Minimal inline copies of alias logic (keep in sync with channelAliases.ts)
const IPTV_ORG_ALIASES = {
  cops: "Cops.us",
  "the pet collective": "ThePetCollective.us",
  "pet collective": "ThePetCollective.us",
  failarmy: "FailArmy.us",
  "red bull tv": "RedBullTV.us",
  "comedy central": "ComedyCentral.us",
  "bbc top gear": "BBCTopGear.uk",
  moviesphere: "MovieSphere.us",
  "movie sphere": "MovieSphere.us",
  "ion mystery": "IonMystery.us",
  "on mystery": "IonMystery.us",
  "wfxt-dt2": "IonMystery.us",
  "wfxt dt2": "IonMystery.us",
  "abc kids": "ABCKids.au",
  "fifa+ united states": "FIFAPlus.uk",
  "fifa+": "FIFAPlus.uk",
  charge: "Charge.us",
  "entertainment tonight": "EntertainmentTonight.us",
  discoverfilm: "DiscoverFilm.uk",
  "discover film": "DiscoverFilm.uk",
  "positiv tv": "PositivTV.us",
  "rally tv": "RallyTV.us",
  "the film detective": "TheFilmDetective.us",
  "amc absolute reality": "AbsoluteRealitybyWETV.us",
  "deluxe music": "DeluxeMusic.de",
  "inwild": "INWILD.nl",
  "360tunebox": "360TuneBox.nl",
  "mix hollywood": "MixHollywood.eg",
  "global fashion channel": "GlobalFashionChannel.us",
  "the chat show channel": "TheChatShowChannel.uk",
  "history hunters": "HistoryHunters.uk",
  "space series": "SpaceSeries.us",
  "whiplash cinema": "WhiplashCinema.us",
  "tvs cipher network": "TVSCipherNetwork.us",
  "tom and jerry": "TomAndJerry.do",
  "30a tv classic movies": "30ATVClassicMovies.us",
  "autentic history": "AutenticHistory.de",
};

const MJH_NAME_HINTS = {
  "Cops.us": ["cops", "cops tv"],
  "ThePetCollective.us": ["the pet collective", "pet collective"],
  "BBCTopGear.uk": ["bbc top gear", "top gear"],
  "RedBullTV.us": ["red bull tv", "redbull tv"],
  "MovieSphere.us": ["moviesphere", "movie sphere"],
  "IonMystery.us": ["ion mystery", "on mystery", "wfxt-dt2", "wfxt dt2", "wfxt 66 2"],
  "WFXT662.us": ["ion mystery", "on mystery", "wfxt-dt2", "wfxt dt2"],
  "ABCKids.au": ["abc kids"],
  "FIFAPlus.uk": ["fifa+", "fifa plus"],
  "Charge.us": ["charge!", "charge"],
  "EntertainmentTonight.us": ["entertainment tonight", "et"],
  "DiscoverFilm.uk": ["discoverfilm", "discover film"],
  "PositivTV.us": ["positiv", "positiv tv"],
  "RallyTV.us": ["rally tv", "rally"],
  "TheFilmDetective.us": ["the film detective", "film detective"],
  "AbsoluteRealitybyWETV.us": ["all reality we tv", "absolute reality", "amc absolute reality"],
  "DeluxeMusic.de": ["deluxe music"],
  "INWILD.nl": ["inwild"],
  "360TuneBox.nl": ["360tunebox", "360 tunebox"],
  "MixHollywood.eg": ["mix hollywood"],
  "GlobalFashionChannel.us": ["global fashion", "fashiontv"],
  "TheChatShowChannel.uk": ["the chat show", "chat show"],
  "HistoryHunters.uk": ["history hunters"],
  "SpaceSeries.us": ["space series", "space & beyond"],
  "WhiplashCinema.us": ["whiplash cinema", "whiplash"],
  "TVSCipherNetwork.us": ["tvs cipher", "cipher network"],
  "TomAndJerry.do": ["tom and jerry", "tom & jerry"],
  "30ATVClassicMovies.us": ["30a tv classic", "30a classic movies"],
  "AutenticHistory.de": ["autentic history"],
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
  if (IPTV_ORG_ALIASES[key]) return IPTV_ORG_ALIASES[key];
  for (const [alias, id] of Object.entries(IPTV_ORG_ALIASES)) {
    if (key.includes(alias) || alias.includes(key)) return IPTV_ORG_ALIASES[alias];
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
  if (orgId) for (const hint of MJH_NAME_HINTS[orgId] ?? []) add(hint);
  if (tvgId?.trim()) {
    const tvgBase = tvgId.includes("@") ? tvgId.split("@")[0].trim() : tvgId.trim();
    for (const hint of MJH_NAME_HINTS[tvgBase] ?? []) add(hint);
  }
  for (const [alias, id] of Object.entries(IPTV_ORG_ALIASES)) {
    if (norm === alias || norm.includes(alias) || alias.includes(norm)) {
      for (const hint of MJH_NAME_HINTS[id] ?? []) add(hint);
    }
  }
  if (/ionmystery/i.test(streamUrl ?? "")) add("ION Mystery");
  if (/abc-kids|abckids/i.test(streamUrl ?? "")) add("ABC Kids");
  if (/fifa/i.test(streamUrl ?? "")) add("FIFA+");
  if (/discoverfilm/i.test(streamUrl ?? "")) add("DiscoverFilm");
  if (/thefilmdetective|film-detective/i.test(streamUrl ?? "")) add("The Film Detective");
  if (/absolutereality|wetv/i.test(streamUrl ?? "")) add("All Reality We TV");
  if (/charge/i.test(streamUrl ?? "")) add("Charge!");
  if (/entertainmenttonight|enterbcef/i.test(streamUrl ?? "")) add("Entertainment Tonight");
  return out;
}

// ... scoreChannelMatch from probe-all-favorites (abbreviated - import logic)
const GENERIC_TOKENS = new Set(["movies","movie","tv","television","channel","live","news","sports","sport","music","plus","the","and","hd","free","international","network","entertainment","family","kids","classic","world","america","american","usa","uk","video","on","demand"]);
const STREAM_PATH_IGNORE = new Set(["master","playlist","index","live","stream","hls","chunklist","manifest","media","pri","sd","hd","m3u8"]);
const MIN_SCORE = 65;
function tokensOf(s){return s.toLowerCase().split(/[^a-z0-9\u0590-\u05ff]+/).filter(Boolean);}
function significantTokens(s){return tokensOf(s).filter(t=>!GENERIC_TOKENS.has(t)&&t.length>=2);}
function nameHasToken(channelName,token){return significantTokens(channelName).includes(token);}
function compactAlpha(s){return s.toLowerCase().replace(/[^a-z0-9]/g,"");}
function stripTvgFeed(tvgId){const at=tvgId.indexOf("@");return at>=0?tvgId.slice(0,at):tvgId;}
function tvgIdVariants(tvgId){if(!tvgId?.trim())return[];const bare=stripTvgFeed(tvgId.trim());const out=[compactAlpha(bare)];const noRegion=bare.replace(/\.[a-z]{2}$/i,"");const compact=compactAlpha(noRegion);if(compact&&!out.includes(compact))out.push(compact);return out.filter(v=>v.length>=4);}
function streamPathTokens(streamUrl){if(!streamUrl)return[];try{const tokens=new Set();for(const seg of new URL(streamUrl).pathname.split("/").filter(Boolean)){const base=seg.replace(/\.m3u8$/i,"");for(const t of significantTokens(base.replace(/_/g," "))){if(!STREAM_PATH_IGNORE.has(t))tokens.add(t);}}return[...tokens];}catch{return[];}}
function scoreChannelMatch(channel,title,tvgId,streamUrl){const normTitle=normalizeForMatch(title);const normName=normalizeForMatch(channel.name);if(normTitle&&normTitle===normName)return 100;const idCompact=compactAlpha(channel.id);const nameCompact=compactAlpha(channel.name);for(const variant of tvgIdVariants(tvgId)){if(idCompact.includes(variant)||nameCompact.includes(variant))return 96;}const titleSig=significantTokens(title);const nameSig=significantTokens(channel.name);const pathSig=streamPathTokens(streamUrl);if(titleSig.length>=2&&titleSig.every(t=>nameHasToken(channel.name,t)))return 85+titleSig.length*4;if(titleSig.length===1&&titleSig[0].length>=6&&nameHasToken(channel.name,titleSig[0]))return 84;if(nameSig.length>=2&&nameSig.every(t=>tokensOf(title).includes(t)))return 80+nameSig.length*4;if(pathSig.length>=2&&pathSig.every(t=>nameHasToken(channel.name,t)))return 78+pathSig.length*4;if(pathSig.length===1&&pathSig[0].length>=6&&nameHasToken(channel.name,pathSig[0]))return 77;const sharedSig=titleSig.filter(t=>nameSig.includes(t));if(sharedSig.length>=2)return 70+sharedSig.length*5;return 0;}
function findBest(channels,title,tvgId,streamUrl){let best=null;for(const ch of channels){const score=scoreChannelMatch(ch,title,tvgId,streamUrl);if(score<MIN_SCORE)continue;if(!best||score>best.score)best={channel:ch,score};}return best;}

const favorites = JSON.parse(readFileSync(join(root, "public/liveMedia/curatedFavorites.json"), "utf8"));
const results = JSON.parse(readFileSync(join(root, "scripts/probe-all-favorites-results.json"), "utf8"));
const noMatch = results.filter(r=>r.status==="NO_MATCH");

const SOURCES = [
  { key: "mjh-all", label: "MJH All", url: "https://i.mjh.nz/all/epg.xml.gz", hint: null },
  { key: "mjh-plex-us", label: "Plex US", url: "https://i.mjh.nz/Plex/us.xml.gz", hint: /plex\.|wurl\.tv|amagi\.tv|mediatailor/i },
  { key: "mjh-pluto-us", label: "Pluto US", url: "https://i.mjh.nz/PlutoTV/us.xml.gz", hint: /pluto\.tv/i },
  { key: "mjh-samsung-us", label: "Samsung US", url: "https://i.mjh.nz/SamsungTVPlus/us.xml.gz", hint: /samsung/i },
  { key: "mjh-roku", label: "Roku", url: "https://i.mjh.nz/Roku/all.xml.gz", hint: /roku/i },
];

function parseChannels(xml, sourceKey) {
  const out = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/g;
  let m;
  while ((m = re.exec(xml))) out.push({ id: m[1], name: m[2], sourceKey });
  return out;
}
function orderedSources(stream) {
  const all = SOURCES[0];
  const hinted = SOURCES.slice(1).filter((s) => s.hint?.test(stream));
  const rest = SOURCES.slice(1).filter((s) => !s.hint?.test(stream));
  return [all, ...hinted, ...rest];
}

const xmlByKey = new Map();
const channelsByKey = new Map();
for (const src of SOURCES) {
  const buf = Buffer.from(await (await fetch(src.url)).arrayBuffer());
  const xml = gunzipSync(buf).toString("utf8");
  xmlByKey.set(src.key, xml);
  channelsByKey.set(src.key, parseChannels(xml, src.key));
}

const fixed = [];
const stillNo = [];

for (const r of noMatch) {
  const ch = favorites.channels.find((c) => c.name === r.name);
  const title = normalizeTitle(ch.name);
  const matchTitles = resolveEpgMatchTitles(title, ch.tvgId, ch.stream);
  const sources = orderedSources(ch.stream);
  let best = null;
  for (const src of sources) {
    const channels = channelsByKey.get(src.key);
    for (const mt of matchTitles) {
      const hit = findBest(channels, mt, ch.tvgId, ch.stream);
      if (!hit) continue;
      const score = scoreChannelMatch(hit.channel, mt, ch.tvgId, ch.stream);
      if (!best || score > best.score) best = { ...hit, score, src: src.label, matchTitle: mt };
    }
  }
  if (best) {
    fixed.push({ name: r.name, ...best, matchTitles });
  } else {
    stillNo.push({ name: r.name, matchTitles });
  }
}

console.log("FIXED with resolveEpgMatchTitles:", fixed.length);
for (const f of fixed) console.log(`  ✓ ${f.name} → ${f.channel.name} [${f.src}] score=${f.score} via "${f.matchTitle}"`);

console.log("\nSTILL NO MATCH:", stillNo.length);
for (const s of stillNo) console.log(`  ✗ ${s.name} | tried: ${s.matchTitles.join(", ")}`);
