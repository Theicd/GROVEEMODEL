import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const results = JSON.parse(readFileSync(join(root, "scripts/probe-all-favorites-results.json"), "utf8"));

const SOURCE_MAP = {
  "Plex US": "mjh-plex-us",
  Roku: "mjh-roku",
  "Samsung US": "mjh-samsung-us",
  "MJH All": "mjh-all",
  "Pluto US": "mjh-pluto-us",
};

const byOrg = {};
for (const x of results) {
  if (x.status !== "OK" || !x.epgId) continue;
  const orgId = x.orgId ?? normalizeOrgFromName(x.name);
  if (!orgId) continue;
  const sourceKey = SOURCE_MAP[x.source];
  if (!sourceKey) continue;
  if (!byOrg[orgId]) byOrg[orgId] = [];
  const channelName = decodeXml(x.epgName ?? "").trim();
  const target = { sourceKey, channelId: x.epgId, ...(channelName ? { channelName } : {}) };
  const key = `${target.sourceKey}|${target.channelId}`;
  if (!byOrg[orgId].some((t) => `${t.sourceKey}|${t.channelId}` === key)) {
    byOrg[orgId].push(target);
  }
}

function decodeXml(s) {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

// Comedy Central: real linear schedule from epg.pw (matches broadcast), Pluto/Roku FAST only as fallback.
byOrg["ComedyCentral.us"] = [
  {
    sourceKey: "epgpw-comedycentral-east",
    channelId: "464922",
    channelName: "Comedy Central HD",
    feedUrl: "https://epg.pw/api/epg.xml?channel_id=464922",
    sourceLabel: "epg.pw",
  },
  { sourceKey: "mjh-pluto-us", channelId: "5ca671f215a62078d2ec0abf", channelName: "Comedy Central Pluto TV" },
  { sourceKey: "mjh-roku", channelId: "3d3f3113ff49ca22c3ad51ee00fe7e9d", channelName: "Comedy Central Pluto TV" },
];

// COPS: verified roku id from probe (mediatailor streams)
byOrg["Cops.us"] = [{ sourceKey: "mjh-roku", channelId: "1a50b7ba48389669b3e0bc6750fe6b31", channelName: "COPS" }];

// Space Series: roku id from probe (not plex duplicate id)
byOrg["SpaceSeries.us"] = [
  { sourceKey: "mjh-roku", channelId: "21520542993cb0e350d453c13a2f3654", channelName: "Space & Beyond" },
  { sourceKey: "mjh-plex-us", channelId: "6a1610bebdf296985fd95603-69d93c22a5c870fb1e88f871", channelName: "Space Series" },
];

function normalizeOrgFromName(name) {
  return null;
}

const lines = [
  'import type { EpgExplicitTarget } from "./epgExplicitBindings";',
  "",
  "/** Probe-verified MJH XMLTV ids keyed by iptv-org channel id. */",
  "export const VERIFIED_EPG_BY_ORG: Record<string, EpgExplicitTarget[]> = {",
];
for (const orgId of Object.keys(byOrg).sort()) {
  lines.push(`  ${JSON.stringify(orgId)}: ${JSON.stringify(byOrg[orgId])},`);
}
lines.push("};", "");

writeFileSync(join(root, "app/src/liveMedia/epg/epgVerifiedCatalog.ts"), lines.join("\n"));
console.log("wrote", Object.keys(byOrg).length, "org bindings");
