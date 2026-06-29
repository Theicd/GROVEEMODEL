import { stripTvgFeed } from "./normalize";

export type EpgExplicitTarget = {
  sourceKey: string;
  /** Fast-path id hint. Samsung/FAST platforms recycle ids, so it is verified against channelName at runtime. */
  channelId?: string;
  /** Canonical XMLTV display-name — used to re-resolve the current id when channelId has rotated. */
  channelName?: string;
  /** Direct per-channel XMLTV feed (e.g. epg.pw API) — used instead of the static source registry. */
  feedUrl?: string;
  /** Display label shown in the OSD when feedUrl is used. */
  sourceLabel?: string;
};

/** Verified MJH XMLTV channel ids — bypass fuzzy match when tvg-id or stream is known. */
const BY_TVG_BASE: Record<string, EpgExplicitTarget[]> = {
  "WFXT662.us": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-62b45f15b4508e0eedacdf26" },
    { sourceKey: "mjh-samsung-us", channelId: "USBC24000223S" },
    { sourceKey: "mjh-roku", channelId: "16f751e2330d5a09a5e1a25a52b2b09c" },
  ],
  "IonMystery.us": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-62b45f15b4508e0eedacdf26" },
    { sourceKey: "mjh-samsung-us", channelId: "USBC24000223S" },
    { sourceKey: "mjh-roku", channelId: "16f751e2330d5a09a5e1a25a52b2b09c" },
  ],
  "FIFAPlus.uk": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-66628d4a8dfc36b8c8a399c4" },
    { sourceKey: "mjh-samsung-us", channelId: "USBD12000255B" },
    { sourceKey: "mjh-roku", channelId: "b8c4d1dc632d5402afce99ee70859b5e" },
  ],
  "AbsoluteRealitybyWETV.us": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-5fc705ff2f220e002d5e6bec" },
    { sourceKey: "mjh-pluto-us", channelId: "5e82530945600e0007ca076c" },
    { sourceKey: "mjh-samsung-us", channelId: "USBD12000061R" },
    { sourceKey: "mjh-roku", channelId: "ebedf88c76db5154b9a747a54d393758" },
  ],
  "EntertainmentTonight.us": [
    { sourceKey: "mjh-roku", channelId: "392a421311b35de594961680be62564c" },
    { sourceKey: "mjh-pluto-us", channelId: "5dc0c78281eddb0009a02d5e" },
    { sourceKey: "mjh-samsung-us", channelId: "USBA3700002JF" },
  ],
  "GlobalFashionChannel.us": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-6490c01f3b3ce9e1aaad95be" },
  ],
  "SpaceSeries.us": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-69d93c22a5c870fb1e88f871" },
    { sourceKey: "mjh-roku", channelId: "6a1610bebdf296985fd95603-69d93c22a5c870fb1e88f871" },
  ],
  "FTFSports.us": [
    { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-605a309dc5acdc002c7a20aa" },
  ],
  "SavedByTheBell.us": [
    { sourceKey: "mjh-roku", channelId: "05a58f8f0d1b55999a9ab0e9caae8a47" },
  ],
  "HistoryHunters.uk": [{ sourceKey: "rakuten-uk", channelId: "history-hunters" }],
  "history-hunters": [{ sourceKey: "rakuten-uk", channelId: "history-hunters" }],
  "MovieSphere.us": [
    { sourceKey: "mjh-samsung-us", channelId: "USBA370000104", channelName: "MovieSphere" },
    { sourceKey: "mjh-samsung-us", channelId: "USBD17000117B", channelName: "MovieSphere" },
    { sourceKey: "mjh-roku", channelId: "e1192d17771c5a059930192d439957ee", channelName: "MovieSphere" },
  ],
  "ABCKids.au": [{ sourceKey: "mjh-all", channelId: "mjh-abc-kids", channelName: "ABC Kids" }],
};

/** Regional tvg-id feeds (@UK, @AU, …) — must win over generic .us base. */
const BY_TVG_FEED: Record<string, EpgExplicitTarget[]> = {
  "MovieSphere.us@UK": [
    { sourceKey: "mjh-samsung-gb", channelId: "GBBA33000557H", channelName: "Moviesphere by Lionsgate" },
  ],
  "MovieSphere.us@US": [{ sourceKey: "mjh-samsung-us", channelId: "USBA370000104", channelName: "MovieSphere" }],
  "MovieSphere.us@CA": [{ sourceKey: "mjh-samsung-ca", channelId: "CA1400004AE", channelName: "MovieSphere" }],
  "ComedyCentral.us@East": [
    // Real linear Comedy Central (East) from epg.pw — matches the actual broadcast (South Park etc.).
    {
      sourceKey: "epgpw-comedycentral-east",
      channelId: "464922",
      channelName: "Comedy Central HD",
      feedUrl: "https://epg.pw/api/epg.xml?channel_id=464922",
      sourceLabel: "epg.pw",
    },
    // Pluto/Roku FAST feeds are a different schedule — kept only as a last-resort fallback.
    { sourceKey: "mjh-pluto-us", channelId: "5ca671f215a62078d2ec0abf", channelName: "Comedy Central Pluto TV" },
    { sourceKey: "mjh-roku", channelId: "3d3f3113ff49ca22c3ad51ee00fe7e9d", channelName: "Comedy Central Pluto TV" },
  ],
};

const STREAM_HINTS: Array<{ test: RegExp; targets: EpgExplicitTarget[] }> = [
  { test: /ftf-linear|ftfsports/i, targets: BY_TVG_BASE["FTFSports.us"]! },
  { test: /nbcuni\.com|savedbythebell|saved-by-the-bell/i, targets: BY_TVG_BASE["SavedByTheBell.us"]! },
  { test: /historyhuntersrakuten|history-hunters/i, targets: BY_TVG_BASE["HistoryHunters.uk"]! },
  { test: /ionmystery/i, targets: BY_TVG_BASE["WFXT662.us"]! },
  { test: /sysdata_s_p_a_fifa|fifa[_-]?plus/i, targets: BY_TVG_BASE["FIFAPlus.uk"]! },
  { test: /absolutereality|wetv/i, targets: BY_TVG_BASE["AbsoluteRealitybyWETV.us"]! },
  { test: /enterbcef|entertainmenttonight/i, targets: BY_TVG_BASE["EntertainmentTonight.us"]! },
  { test: /globalfashionchannel|pubgfc/i, targets: BY_TVG_BASE["GlobalFashionChannel.us"]! },
  { test: /abc-kids|abckids/i, targets: BY_TVG_BASE["ABCKids.au"]! },
  { test: /moviesphereuk-samsunguk|moviesphereuk\.amagi/i, targets: BY_TVG_FEED["MovieSphere.us@UK"]! },
  { test: /usa_comedy_central/i, targets: BY_TVG_FEED["ComedyCentral.us@East"]! },
];

function dedupeTargets(targets: EpgExplicitTarget[]): EpgExplicitTarget[] {
  const seen = new Set<string>();
  const out: EpgExplicitTarget[] = [];
  for (const t of targets) {
    const key = `${t.sourceKey}|${t.channelId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(t);
  }
  return out;
}

import { israelEpgTargets } from "./israelEpgBindings";
import { VERIFIED_EPG_BY_ORG } from "./epgVerifiedCatalog";

/**
 * Ordered explicit XMLTV targets for a favorite (highest confidence first).
 * `orgId` lets channels without a tvg-id (resolved from the title alias) still use the verified catalog.
 */
export function explicitEpgTargets(tvgId?: string, streamUrl?: string, orgId?: string): EpgExplicitTarget[] {
  const out: EpgExplicitTarget[] = [];
  const raw = tvgId?.trim() ?? "";
  if (raw && BY_TVG_FEED[raw]) out.push(...BY_TVG_FEED[raw]);
  const base = raw ? stripTvgFeed(raw) : "";
  if (base && BY_TVG_BASE[base]) out.push(...BY_TVG_BASE[base]);
  if (base && VERIFIED_EPG_BY_ORG[base]) out.push(...VERIFIED_EPG_BY_ORG[base]);
  const org = orgId?.trim();
  if (org && org !== base) {
    if (BY_TVG_BASE[org]) out.push(...BY_TVG_BASE[org]);
    if (VERIFIED_EPG_BY_ORG[org]) out.push(...VERIFIED_EPG_BY_ORG[org]);
  }
  out.push(...israelEpgTargets(tvgId));
  for (const hint of STREAM_HINTS) {
    if (hint.test.test(streamUrl ?? "")) out.push(...hint.targets);
  }
  return dedupeTargets(out);
}
