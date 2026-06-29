import type { UnifiedSearchHit } from "../../searchResults/types";
import { channelMayHaveEpg, resolveEpgMatchTitles, resolveIptvOrgChannelId } from "./channelAliases";
import { findBestChannelMatch, scoreChannelMatch } from "./channelMatch";
import { explicitEpgTargets, type EpgExplicitTarget } from "./epgExplicitBindings";
import { streamEpgAffinityBonus } from "./epgStreamAffinity";
import type { EpgChannelRef } from "./types";
import { isIsraelTvgId } from "./israelEpgBindings";
import { fetchMjhXmltv, findEpgSourceByKey, MJH_EPG_SOURCES, orderedSourcesForStream, warmMjhEpgCaches } from "./mjhSources";
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

const EPG_MATCH_VERSION = 4;
const availabilityCache = new Map<string, boolean>();

function lookupCacheKey(input: EpgLookupInput): string {
  return `v${EPG_MATCH_VERSION}|${normalizeForMatch(input.title)}|${input.tvgId ?? ""}|${input.streamUrl ?? ""}`;
}

export function hitToEpgLookup(hit: UnifiedSearchHit | null): EpgLookupInput | null {
  if (!hit || (hit.kind !== "livetv" && hit.kind !== "youtube")) return null;
  const epgTitle =
    typeof hit.meta?.epgTitle === "string" && hit.meta.epgTitle
      ? hit.meta.epgTitle
      : hit.titleOriginal || hit.title;
  return {
    title: epgTitle,
    streamUrl: hit.mediaPlayUrl || hit.url,
    tvgId: typeof hit.meta?.tvgId === "string" ? hit.meta.tvgId : undefined,
  };
}

/** iptv-org ids for regions without a public MJH/XMLTV programme feed. */
function lacksOpenProgrammeFeed(orgId: string | null): boolean {
  return orgId != null && /\.il$/i.test(orgId);
}

/**
 * Resolve an explicit target to a live channel in the feed.
 * Samsung/FAST platforms recycle ids, so a hardcoded channelId can point at the wrong
 * channel later. When channelName is known we verify the id still matches that name and,
 * if not, re-resolve by name so the binding self-heals as ids rotate.
 */
export function resolveExplicitChannel(
  channels: EpgChannelRef[],
  target: EpgExplicitTarget,
): EpgChannelRef | null {
  const wantName = target.channelName ? normalizeForMatch(target.channelName) : "";

  if (target.channelId) {
    const byId = channels.find((c) => c.id === target.channelId);
    if (byId && (!wantName || normalizeForMatch(byId.name) === wantName)) return byId;
  }

  if (wantName) {
    const exact = channels.find((c) => normalizeForMatch(c.name) === wantName);
    if (exact) return exact;
    const contains = channels.find((c) => {
      const n = normalizeForMatch(c.name);
      return n.length >= 4 && (n.includes(wantName) || wantName.includes(n));
    });
    if (contains) return contains;
  }

  if (target.channelId && !wantName) {
    return channels.find((c) => c.id === target.channelId) ?? null;
  }
  return null;
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
  const israelOnly = lacksOpenProgrammeFeed(orgId) || isIsraelTvgId(input.tvgId);

  const stream = input.streamUrl ?? "";
  const sources = orderedSourcesForStream(stream);
  const matchTitles = israelOnly ? [] : resolveEpgMatchTitles(title, input.tvgId, stream);

  await warmMjhEpgCaches(stream);

  let best: {
    channel: NonNullable<ReturnType<typeof findBestChannelMatch>>;
    xml: string;
    source: (typeof MJH_EPG_SOURCES)[number];
    score: number;
  } | null = null;

  const tryCandidate = (
    channel: NonNullable<ReturnType<typeof findBestChannelMatch>>,
    xml: string,
    source: { key: string; label: string; url: string },
    score: number,
  ) => {
    if (!best || score > best.score) best = { channel, xml, source, score };
  };

  for (const target of explicitEpgTargets(input.tvgId, stream, orgId ?? undefined)) {
    const source = target.feedUrl
      ? { key: target.sourceKey, label: target.sourceLabel ?? "EPG", url: target.feedUrl }
      : findEpgSourceByKey(target.sourceKey);
    if (!source) continue;
    const xml = await fetchMjhXmltv(source.url);
    if (!xml) continue;
    const channels = parsedChannelsForXml(source.key, xml);
    const channel = resolveExplicitChannel(channels, target);
    if (!channel) continue;
    if (!xmlHasProgrammesForChannel(xml, channel.id)) continue;

    if (opts?.probeOnly) {
      tryCandidate(channel, xml, source, 1000);
      continue;
    }

    const allPrograms = parseXmltvPrograms(xml, channel.id);
    const programs = opts?.guide
      ? pickProgramsForGuide(allPrograms)
      : pickCurrentAndUpcoming(allPrograms);
    const usable = programs.length > 0 ? programs : allPrograms.slice(0, 48);
    if (usable.length > 0) tryCandidate(channel, xml, source, 1000);
  }

  // Explicit bindings are authoritative — skip the expensive fuzzy scan when one resolved.
  if (best) {
    return finalizeSchedule(best, opts);
  }

  for (const source of sources) {
    const xml = await fetchMjhXmltv(source.url);
    if (!xml) continue;

    const channels = parsedChannelsForXml(source.key, xml);
    for (const matchTitle of matchTitles) {
      const channel = findBestChannelMatch(channels, matchTitle, input.tvgId, stream);
      if (!channel) continue;

      const baseScore = scoreChannelMatch(channel, matchTitle, input.tvgId, stream);
      const score =
        baseScore + streamEpgAffinityBonus(stream, channel.name, source.key);
      if (score < 65) continue;
      tryCandidate(channel, xml, source, score);
    }
  }

  return finalizeSchedule(best, opts);
}

type BestCandidate = {
  channel: NonNullable<ReturnType<typeof findBestChannelMatch>>;
  xml: string;
  source: { key: string; label: string; url: string };
  score: number;
};

function finalizeSchedule(
  best: BestCandidate | null,
  opts?: { probeOnly?: boolean; guide?: boolean },
): EpgSchedule | null {
  if (!best) return null;

  if (opts?.probeOnly) {
    if (xmlHasProgrammesForChannel(best.xml, best.channel.id)) {
      return { channel: best.channel, programs: [], sourceLabel: best.source.label };
    }
    return null;
  }

  const allPrograms = parseXmltvPrograms(best.xml, best.channel.id);
  const windowed = opts?.guide ? pickProgramsForGuide(allPrograms) : pickCurrentAndUpcoming(allPrograms);
  const programs = windowed.length > 0 ? windowed : allPrograms.slice(0, 48);
  if (programs.length > 0) {
    return { channel: best.channel, programs, sourceLabel: best.source.label };
  }

  return null;
}
