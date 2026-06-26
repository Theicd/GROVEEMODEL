import { normalizeChannelTitle, normalizeForMatch, stripTvgFeed } from "./normalize";

/** Favorite / IPTV display name → iptv-org database channel id (for guides.json lookup). */
const IPTV_ORG_ALIASES: Record<string, string> = {
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
  "kan 11": "Kan11.il",
  "channel 9": "Channel9.il",
  channel9: "Channel9.il",
  "design network": "TheDesignNetwork.us",
  "wild earth": "WildEarth.us",
  wildearth: "WildEarth.us",
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
  inwild: "INWILD.nl",
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

/** Extra XMLTV display-name keys to try after the human title. */
const MJH_NAME_HINTS: Record<string, string[]> = {
  "Cops.us": ["cops", "cops tv"],
  "ThePetCollective.us": ["the pet collective", "pet collective"],
  "BBCTopGear.uk": ["bbc top gear", "top gear"],
  "RedBullTV.us": ["red bull tv", "redbull tv"],
  "MovieSphere.us": ["moviesphere", "movie sphere"],
  "IonMystery.us": ["ion mystery", "on mystery", "wfxt-dt2", "wfxt dt2", "wfxt 66 2"],
  "WFXT662.us": ["ion mystery", "on mystery", "wfxt-dt2", "wfxt dt2"],
  "ABCKids.au": ["abc kids"],
  "FIFAPlus.uk": ["FIFA+", "fifa+", "fifa plus"],
  "Charge.us": ["charge!", "charge"],
  "EntertainmentTonight.us": ["Entertainment Tonight", "entertainment tonight", "ET"],
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

export function resolveIptvOrgChannelId(title: string, tvgId?: string): string | null {
  if (tvgId?.trim()) {
    const base = tvgId.includes("@") ? tvgId.split("@")[0] : tvgId;
    if (base.trim()) return base.trim();
  }
  const key = normalizeForMatch(title);
  if (IPTV_ORG_ALIASES[key]) return IPTV_ORG_ALIASES[key];
  for (const [alias, id] of Object.entries(IPTV_ORG_ALIASES)) {
    if (key.includes(alias) || alias.includes(key)) return id;
  }
  return null;
}

/** Ordered unique keys for matching a channel inside XMLTV feeds. */
export function resolveMatchKeys(title: string, tvgId?: string): string[] {
  const keys: string[] = [];
  const add = (value: string | null | undefined) => {
    const v = value?.trim();
    if (!v) return;
    const norm = normalizeForMatch(v);
    if (!norm) return;
    if (!keys.includes(norm)) keys.push(norm);
    if (!keys.includes(v.toLowerCase())) keys.push(v.toLowerCase());
  };

  add(normalizeForMatch(title));
  add(title);
  const orgId = resolveIptvOrgChannelId(title, tvgId);
  add(orgId);
  if (orgId) {
    for (const hint of MJH_NAME_HINTS[orgId] ?? []) add(hint);
  }
  return keys;
}

/** Alternate display titles to try when matching XMLTV (mislabeled IPTV names). */
export function resolveEpgMatchTitles(title: string, tvgId?: string, streamUrl?: string): string[] {
  const out: string[] = [];
  const add = (value: string | null | undefined) => {
    const v = value?.trim();
    if (!v) return;
    const norm = normalizeChannelTitle(v);
    if (norm && !out.includes(norm)) out.push(norm);
  };

  add(title);
  const norm = normalizeForMatch(title);
  const orgId = resolveIptvOrgChannelId(title, tvgId);
  if (orgId) {
    for (const hint of MJH_NAME_HINTS[orgId] ?? []) add(hint);
  }
  if (tvgId?.trim()) {
    const tvgBase = tvgId.includes("@") ? tvgId.split("@")[0]!.trim() : tvgId.trim();
    for (const hint of MJH_NAME_HINTS[tvgBase] ?? []) add(hint);
  }
  for (const [alias, id] of Object.entries(IPTV_ORG_ALIASES)) {
    if (norm === alias || norm.includes(alias) || alias.includes(norm)) {
      for (const hint of MJH_NAME_HINTS[id] ?? []) add(hint);
    }
  }
  if (/ionmystery/i.test(streamUrl ?? "")) add("ION Mystery");
  if (/abc-kids|abckids/i.test(streamUrl ?? "")) add("ABC Kids");
  if (/sysdata_s_p_a_fifa|fifa[_-]?plus/i.test(streamUrl ?? "")) add("FIFA+");
  if (/discoverfilm/i.test(streamUrl ?? "")) add("DiscoverFilm");
  if (/thefilmdetective|film-detective/i.test(streamUrl ?? "")) add("The Film Detective");
  if (/absolutereality|wetv/i.test(streamUrl ?? "")) add("All Reality We TV");
  if (/fast-channels\.sinclairstoryline\.com\/charge|\/charge\//i.test(streamUrl ?? "")) add("Charge!");
  if (/enterbcef|entertainmenttonight/i.test(streamUrl ?? "")) add("ET");
  if (/historyhuntersrakuten|history-hunters/i.test(streamUrl ?? "")) add("History Hunters");
  if (/globalfashionchannel|pubgfc/i.test(streamUrl ?? "")) add("FashionTV");
  if (/sofast\.tv/i.test(streamUrl ?? "")) add("Space & Beyond");
  return out;
}

/** Instant hint for UI — true for curated / MJH streams before the full probe finishes. */
export function channelMayHaveEpg(title: string, tvgId?: string, streamUrl?: string): boolean {
  const norm = normalizeForMatch(title);
  const orgId = resolveIptvOrgChannelId(title, tvgId);
  if (orgId && /\.il$/i.test(orgId)) return false;

  if (orgId && MJH_NAME_HINTS[orgId]) return true;
  if (IPTV_ORG_ALIASES[norm] && !/\.il$/i.test(IPTV_ORG_ALIASES[norm])) return true;
  for (const [alias, id] of Object.entries(IPTV_ORG_ALIASES)) {
    if ((norm.includes(alias) || alias.includes(norm)) && !/\.il$/i.test(id)) return true;
  }

  const bareTvg = tvgId?.trim() ? stripTvgFeed(tvgId.trim()) : "";
  if (bareTvg && !/\.il$/i.test(bareTvg)) return true;

  if (/c\.mjh\.nz|pluto\.tv|plex\.|samsungtvplus|roku\.com|amagi\.tv|wurl\.tv/i.test(streamUrl ?? "")) {
    return true;
  }
  return false;
}
