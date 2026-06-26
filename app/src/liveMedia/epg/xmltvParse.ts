import { findBestChannelMatch } from "./channelMatch";
import type { EpgChannelRef, EpgProgram } from "./types";

function parseXmltvDate(raw: string): Date | null {
  const m = raw.trim().match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\s*([+-]\d{4})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s, tz] = m;
  const iso = `${y}-${mo}-${d}T${h}:${mi}:${s}${tz ? `${tz.slice(0, 3)}:${tz.slice(3)}` : "Z"}`;
  const dt = new Date(iso);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

export function parseXmltvChannels(xml: string, sourceKey: string): EpgChannelRef[] {
  return parseXmltvChannelsFast(xml, sourceKey);
}

/** Regex channel list — DOMParser on multi-MB XMLTV hangs in the browser. */
export function parseXmltvChannelsFast(xml: string, sourceKey: string): EpgChannelRef[] {
  const out: EpgChannelRef[] = [];
  const re = /<channel id="([^"]+)"[^>]*>[\s\S]*?<display-name>([^<]*)<\/display-name>/gi;
  let m;
  while ((m = re.exec(xml))) {
    out.push({ id: m[1], name: decodeXmlText(m[2]) || m[1], sourceKey });
  }
  return out;
}

export function parseXmltvPrograms(xml: string, channelId: string): EpgProgram[] {
  return parseXmltvProgramsFast(xml, channelId);
}

function decodeXmlText(raw: string): string {
  return raw
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .trim();
}

function tagText(block: string, tag: string): string | undefined {
  const m = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, "i"));
  return m ? decodeXmlText(m[1]) : undefined;
}

function parseLengthMinutes(block: string): number | undefined {
  const m = block.match(/<length([^>]*)>([^<]*)<\/length>/i);
  if (!m) return undefined;
  const raw = m[2].trim();
  const n = Number.parseFloat(raw);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  const units = (m[1].match(/units="([^"]+)"/i)?.[1] ?? "minutes").toLowerCase();
  if (units.startsWith("sec")) return n / 60;
  if (units.startsWith("hour")) return n * 60;
  return n;
}

function parseEpisodeFields(block: string): Pick<EpgProgram, "season" | "episode" | "episodeLabel" | "subTitle"> {
  const subTitle = tagText(block, "sub-title");
  const onscreen = block
    .match(/<episode-num[^>]*system="onscreen"[^>]*>([^<]*)<\/episode-num>/i)?.[1]
    ?.trim();
  if (onscreen) {
    const m = onscreen.match(/S(\d+)\s*E(\d+)/i) ?? onscreen.match(/(\d+)\s*x\s*(\d+)/i);
    if (m) {
      return {
        season: +m[1],
        episode: +m[2],
        episodeLabel: onscreen,
        subTitle,
      };
    }
    return { episodeLabel: onscreen, subTitle };
  }
  const xmltv = block
    .match(/<episode-num[^>]*system="xmltv_ns"[^>]*>([^<]*)<\/episode-num>/i)?.[1]
    ?.trim();
  if (xmltv) {
    const m = xmltv.match(/(\d+)\.(\d+)/);
    if (m) {
      const season = +m[1] + 1;
      const episode = +m[2] + 1;
      return {
        season,
        episode,
        episodeLabel: `S${String(season).padStart(2, "0")}E${String(episode).padStart(2, "0")}`,
        subTitle,
      };
    }
  }
  return { subTitle };
}

/** Regex parser — DOMParser on multi-MB XMLTV hangs in the browser. */
export function parseXmltvProgramsFast(xml: string, channelId: string): EpgProgram[] {
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`<programme\\s([^>]*?)>([\\s\\S]*?)</programme>`, "gi");
  const out: EpgProgram[] = [];
  let m;
  while ((m = re.exec(xml))) {
    const attrs = m[1];
    if (!new RegExp(`channel="${escaped}"`, "i").test(attrs)) continue;
    const start = parseXmltvDate(attrs.match(/start="([^"]+)"/i)?.[1] ?? "");
    const end = parseXmltvDate(attrs.match(/stop="([^"]+)"/i)?.[1] ?? "");
    if (!start || !end) continue;
    const block = m[2];
    out.push({
      channelId,
      title: tagText(block, "title") || "—",
      description: tagText(block, "desc"),
      category: tagText(block, "category"),
      poster: block.match(/<icon[^>]*src="([^"]+)"/i)?.[1],
      start,
      end,
      lengthMinutes: parseLengthMinutes(block),
      ...parseEpisodeFields(block),
    });
  }
  out.sort((a, b) => a.start.getTime() - b.start.getTime());
  return out;
}

const parsedChannelsBySource = new Map<string, EpgChannelRef[]>();

/** Parse channel list once per source — large XMLTV files are expensive to re-parse. */
export function parsedChannelsForXml(sourceKey: string, xml: string): EpgChannelRef[] {
  const cached = parsedChannelsBySource.get(sourceKey);
  if (cached) return cached;
  const list = parseXmltvChannels(xml, sourceKey);
  parsedChannelsBySource.set(sourceKey, list);
  return list;
}

export function resetParsedChannelCacheForTests(): void {
  parsedChannelsBySource.clear();
}

/** Match against an already-parsed channel list (avoids re-parsing large XML per key). */
export function matchChannelInList(
  channels: EpgChannelRef[],
  title: string,
  _matchKey?: string,
  tvgId?: string,
  streamUrl?: string,
): EpgChannelRef | null {
  if (!channels.length) return null;
  return findBestChannelMatch(channels, title, tvgId, streamUrl);
}

/** Find best channel row in parsed XML for a human title. */
export function matchChannelInXml(
  xml: string,
  sourceKey: string,
  title: string,
  matchKey?: string,
  tvgId?: string,
  streamUrl?: string,
): EpgChannelRef | null {
  return matchChannelInList(parseXmltvChannels(xml, sourceKey), title, matchKey, tvgId, streamUrl);
}

/** Fast check — any programme row exists for this channel id. */
export function xmlHasProgrammesForChannel(xml: string, channelId: string): boolean {
  const escaped = channelId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`<programme\\s[^>]*channel="${escaped}"`, "i").test(xml);
}

export function pickProgramsForGuide(programs: EpgProgram[], hoursAhead = 6, now = new Date()): EpgProgram[] {
  const horizon = now.getTime() - 30 * 60_000;
  const end = now.getTime() + hoursAhead * 60 * 60_000;
  return programs.filter((p) => p.end.getTime() > horizon && p.start.getTime() < end);
}

export function pickCurrentAndUpcoming(programs: EpgProgram[], now = new Date(), limit = 48): EpgProgram[] {
  const horizon = now.getTime() - 6 * 60 * 60 * 1000;
  return programs.filter((p) => p.end.getTime() > horizon).slice(0, limit);
}
