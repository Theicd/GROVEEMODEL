import { normalizeForMatch, stripTvgFeed } from "./normalize";
import type { EpgChannelRef } from "./types";

/** Too generic for standalone channel matching. */
const GENERIC_TOKENS = new Set([
  "movies",
  "movie",
  "tv",
  "television",
  "channel",
  "live",
  "news",
  "sports",
  "sport",
  "music",
  "plus",
  "the",
  "and",
  "hd",
  "free",
  "international",
  "network",
  "entertainment",
  "family",
  "kids",
  "classic",
  "world",
  "america",
  "american",
  "usa",
  "uk",
  "video",
  "on",
  "demand",
]);

/** Common HLS path segments — not channel identity. */
const STREAM_PATH_IGNORE = new Set([
  "master",
  "playlist",
  "index",
  "live",
  "stream",
  "hls",
  "chunklist",
  "manifest",
  "media",
  "pri",
  "sd",
  "hd",
  "m3u8",
]);

export const MIN_CHANNEL_MATCH_SCORE = 65;

function tokensOf(s: string): string[] {
  return s
    .toLowerCase()
    .split(/[^a-z0-9\u0590-\u05ff]+/)
    .filter((t) => t.length > 0);
}

function significantTokens(s: string): string[] {
  return tokensOf(s).filter((t) => !GENERIC_TOKENS.has(t) && t.length >= 2);
}

function nameHasToken(channelName: string, token: string): boolean {
  return significantTokens(channelName).includes(token);
}

function compactAlpha(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function tvgIdVariants(tvgId?: string): string[] {
  if (!tvgId?.trim()) return [];
  const bare = stripTvgFeed(tvgId.trim());
  const out = [compactAlpha(bare)];
  const noRegion = bare.replace(/\.[a-z]{2}$/i, "");
  const compact = compactAlpha(noRegion);
  if (compact && !out.includes(compact)) out.push(compact);
  return out.filter((v) => v.length >= 4);
}

function streamPathTokens(streamUrl?: string): string[] {
  if (!streamUrl) return [];
  try {
    const tokens = new Set<string>();
    for (const seg of new URL(streamUrl).pathname.split("/").filter(Boolean)) {
      const base = seg.replace(/\.m3u8$/i, "");
      for (const t of significantTokens(base.replace(/_/g, " "))) {
        if (!STREAM_PATH_IGNORE.has(t)) tokens.add(t);
      }
    }
    return [...tokens];
  } catch {
    return [];
  }
}

function streamPathCompact(streamUrl?: string): string {
  if (!streamUrl) return "";
  try {
    return compactAlpha(new URL(streamUrl).pathname);
  } catch {
    return "";
  }
}

/** Higher = more confident match. */
export function scoreChannelMatch(
  channel: EpgChannelRef,
  title: string,
  tvgId?: string,
  streamUrl?: string,
): number {
  const normTitle = normalizeForMatch(title);
  const normName = normalizeForMatch(channel.name);
  let raw = 0;
  if (normTitle && normTitle === normName) raw = 100;
  else {
    const idCompact = compactAlpha(channel.id);
    const nameCompact = compactAlpha(channel.name);

    for (const variant of tvgIdVariants(tvgId)) {
      if (idCompact.includes(variant) || nameCompact.includes(variant)) {
        raw = 96;
        break;
      }
    }

    if (!raw) {
      const titleSig = significantTokens(title);
      const nameSig = significantTokens(channel.name);
      const pathSig = streamPathTokens(streamUrl);

      if (titleSig.length >= 2 && titleSig.every((t) => nameHasToken(channel.name, t))) {
        raw = 85 + titleSig.length * 4;
      } else if (titleSig.length === 1 && titleSig[0].length >= 6 && nameHasToken(channel.name, titleSig[0])) {
        raw = 84;
      } else if (nameSig.length >= 2 && nameSig.every((t) => tokensOf(title).includes(t))) {
        raw = 80 + nameSig.length * 4;
      } else if (pathSig.length >= 2 && pathSig.every((t) => nameHasToken(channel.name, t))) {
        raw = 78 + pathSig.length * 4;
      } else if (pathSig.length === 1 && pathSig[0].length >= 6 && nameHasToken(channel.name, pathSig[0])) {
        raw = 77;
      } else {
        const pathCompact = streamPathCompact(streamUrl);
        if (nameCompact.length >= 6 && pathCompact.includes(nameCompact)) raw = 86;
        else {
          const sharedSig = titleSig.filter((t) => nameSig.includes(t));
          if (sharedSig.length >= 2) raw = 70 + sharedSig.length * 5;
        }
      }
    }
  }

  return raw;
}

export function findBestChannelMatch(
  channels: EpgChannelRef[],
  title: string,
  tvgId?: string,
  streamUrl?: string,
): EpgChannelRef | null {
  let best: { channel: EpgChannelRef; score: number } | null = null;
  for (const channel of channels) {
    const score = scoreChannelMatch(channel, title, tvgId, streamUrl);
    if (score < MIN_CHANNEL_MATCH_SCORE) continue;
    if (!best || score > best.score) best = { channel, score };
  }
  return best?.channel ?? null;
}
