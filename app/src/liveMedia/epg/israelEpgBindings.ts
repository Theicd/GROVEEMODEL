import type { EpgExplicitTarget } from "./epgExplicitBindings";

/** iptv-org.github.io Israel site guides — xmltv_id → source file + channel id in that file. */
export const ISRAEL_EPG_SOURCES = [
  {
    key: "il-kan",
    label: "Kan Israel",
    url: "https://iptv-org.github.io/epg/guides/il/kan.org.il.epg.xml",
  },
  {
    key: "il-mako",
    label: "Keshet 12",
    url: "https://iptv-org.github.io/epg/guides/il/mako.co.il.epg.xml",
  },
  {
    key: "il-9tv",
    label: "Channel 9",
    url: "https://iptv-org.github.io/epg/guides/il/9tv.co.il.epg.xml",
  },
  {
    key: "il-i24",
    label: "i24 News",
    url: "https://iptv-org.github.io/epg/guides/il/i24news.tv.epg.xml",
  },
] as const;

/** iptv-org channel id (base, no @feed) → explicit XMLTV targets. */
const ISRAEL_TVG_BINDINGS: Record<string, EpgExplicitTarget[]> = {
  "Kan11.il": [{ sourceKey: "il-kan", channelId: "Kan11.il" }],
  "KanEducational.il": [{ sourceKey: "il-kan", channelId: "KanEducational.il" }],
  "Makan33.il": [{ sourceKey: "il-kan", channelId: "Makan33.il" }],
  "Channel12.il": [{ sourceKey: "il-mako", channelId: "Keshet12.il" }],
  "Keshet12.il": [{ sourceKey: "il-mako", channelId: "Keshet12.il" }],
  "Channel9.il": [{ sourceKey: "il-9tv", channelId: "Channel9.il" }],
};/** Reshet 13 / Now 14: no stable public XMLTV feed yet (iptv-org guides often 404). */
export function israelEpgTargets(tvgId?: string): EpgExplicitTarget[] {
  if (!tvgId?.trim()) return [];
  const base = tvgId.includes("@") ? tvgId.split("@")[0]!.trim() : tvgId.trim();
  return ISRAEL_TVG_BINDINGS[base] ?? [];
}

export function isIsraelTvgId(tvgId?: string): boolean {
  if (!tvgId?.trim()) return false;
  const base = tvgId.includes("@") ? tvgId.split("@")[0]! : tvgId;
  return /\.il$/i.test(base);
}
