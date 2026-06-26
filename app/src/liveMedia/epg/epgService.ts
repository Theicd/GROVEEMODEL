import type { UnifiedSearchHit } from "../../searchResults/types";
import { channelMayHaveEpg, resolveEpgMatchTitles, resolveIptvOrgChannelId } from "./channelAliases";
import { findBestChannelMatch, scoreChannelMatch } from "./channelMatch";
import { explicitEpgTargets } from "./epgExplicitBindings";
import { fetchMjhXmltv, MJH_EPG_SOURCES, orderedSourcesForStream, warmMjhEpgCaches } from "./mjhSources";
import { normalizeChannelTitle, normalizeForMatch } from "./normalize";
import type { EpgSchedule } from "./types";
import {
  parsedChannelsForXml,
  parseXmltvPrograms,
  pickCurrentAndUpcoming,
  pickProgramsForGuide,
  resetParsedChannelCacheForTests,
  xmlHasProgrammesForChannel,
} from "./xmltvParse";

export type EpgLookupInput = {
  title: string;
  streamUrl?: string;
  tvgId?: string;
};

const EPG_MATCH_VERSION = 3;
const availabilityCache = new Map<string, boolean>();

function lookupCacheKey(input: EpgLookupInput): string {
  return `v${EPG_MATCH_VERSION}|${normalizeForMatch(input.title)}|${input.tvgId ?? ""}|${input.streamUrl ?? ""}`;
}

export function hitToEpgLookup(hit: UnifiedSearchHit | null): EpgLookupInput | null {
  if (!hit || (hit.kind !== "livetv" && hit.kind !== "youtube")) return null;
  return {
    title: hit.title,
    streamUrl: hit.mediaPlayUrl || hit.url,
    tvgId: typeof hit.meta?.tvgId === "string" ? hit.meta.tvgId : undefined,
  };
}

/** iptv-org ids for regions without a public MJH/XMLTV programme feed. */
function lacksOpenProgrammeFeed(orgId: string | null): boolean {
  return orgId != null && /\.il$/i.test(orgId);
}

export { channelMayHaveEpg, warmMjhEpgCaches };

export async function channelHasEpg(input: EpgLookupInput): Promise<boolean> {
  const key = lookupCacheKey(input);
  const cached = availabilityCache.get(key);
  if (cached != null) return cached;

  const schedule = await fetchEpgSchedule(input, { probeOnly: true });
  const ok = schedule != null;
  availabilityCache.set(key, ok);
  return ok;
}

export function resetEpgAvailabilityCacheForTests(): void {
  availabilityCache.clear();
}

export async function fetchEpgSchedule(
  input: EpgLookupInput,
  opts?: { probeOnly?: boolean; guide?: boolean },
): Promise<EpgSchedule | null> {
  const title = normalizeChannelTitle(input.title);
  const orgId = resolveIptvOrgChannelId(title, input.tvgId);
  if (lacksOpenProgrammeFeed(orgId)) return null;

  const stream = input.streamUrl ?? "";
  const sources = orderedSourcesForStream(stream);
  const matchTitles = resolveEpgMatchTitles(title, input.tvgId, stream);

  await warmMjhEpgCaches(stream);

  let best: {
    channel: NonNullable<ReturnType<typeof findBestChannelMatch>>;
    xml: string;
    source: (typeof sources)[number];
    score: number;
  } | null = null;

  const tryCandidate = (
    channel: NonNullable<ReturnType<typeof findBestChannelMatch>>,
    xml: string,
    source: (typeof MJH_EPG_SOURCES)[number],
    score: number,
  ) => {
    if (!best || score > best.score) best = { channel, xml, source, score };
  };

  for (const target of explicitEpgTargets(input.tvgId, stream)) {
    const source = MJH_EPG_SOURCES.find((s) => s.key === target.sourceKey);
    if (!source) continue;
    const xml = await fetchMjhXmltv(source.url);
    if (!xml) continue;
    const channels = parsedChannelsForXml(source.key, xml);
    const channel = channels.find((c) => c.id === target.channelId);
    if (!channel) continue;
    if (opts?.probeOnly) {
      if (xmlHasProgrammesForChannel(xml, channel.id)) tryCandidate(channel, xml, source, 100);
    } else {
      const programs = opts?.guide
        ? pickProgramsForGuide(parseXmltvPrograms(xml, channel.id))
        : pickCurrentAndUpcoming(parseXmltvPrograms(xml, channel.id));
      if (programs.length > 0) tryCandidate(channel, xml, source, 100);
    }
  }

  for (const source of sources) {
    const xml = await fetchMjhXmltv(source.url);
    if (!xml) continue;

    const channels = parsedChannelsForXml(source.key, xml);
    for (const matchTitle of matchTitles) {
      const channel = findBestChannelMatch(channels, matchTitle, input.tvgId, stream);
      if (!channel) continue;

      const score = scoreChannelMatch(channel, matchTitle, input.tvgId, stream);
      tryCandidate(channel, xml, source, score);
    }
  }

  if (!best) return null;

  if (opts?.probeOnly) {
    if (xmlHasProgrammesForChannel(best.xml, best.channel.id)) {
      return { channel: best.channel, programs: [], sourceLabel: best.source.label };
    }
    return null;
  }

  const programs = opts?.guide
    ? pickProgramsForGuide(parseXmltvPrograms(best.xml, best.channel.id))
    : pickCurrentAndUpcoming(parseXmltvPrograms(best.xml, best.channel.id));
  if (programs.length > 0) {
    return { channel: best.channel, programs, sourceLabel: best.source.label };
  }

  return null;
}
