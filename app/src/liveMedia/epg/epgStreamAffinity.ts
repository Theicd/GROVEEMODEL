/** Stream URL ↔ MJH XMLTV source affinity — boosts correct platform matches, never blocks. */

export type StreamPlatform = "pluto" | "roku" | "samsung" | "plex" | "amagi" | "wurl" | "linear" | "unknown";

export function detectStreamPlatform(streamUrl?: string): StreamPlatform {
  const u = streamUrl ?? "";
  if (/pluto\.tv/i.test(u)) return "pluto";
  if (/roku\.com/i.test(u)) return "roku";
  if (/samsungtvplus|samsunguk|samsungau|samsungca|samsungus/i.test(u)) return "samsung";
  if (/plex\.(tv|wurl)|wurl\.tv|wurl\.com/i.test(u)) return "plex";
  if (/amagi\.tv|mediatailor/i.test(u)) return "amagi";
  if (/wurl/i.test(u)) return "wurl";
  if (/:8080\/|:\/\/\d{1,3}(?:\.\d{1,3}){3}\/|usa_[a-z0-9_]+/i.test(u)) return "linear";
  return "unknown";
}

export function detectChannelPlatform(channelName: string, sourceKey?: string): StreamPlatform {
  const name = channelName.toLowerCase();
  if (/pluto\s*tv/i.test(name)) return "pluto";
  if (sourceKey === "mjh-pluto-us") return "pluto";
  if (sourceKey === "mjh-roku") return "roku";
  if (/samsung/i.test(name) || /^mjh-samsung-/i.test(sourceKey ?? "")) return "samsung";
  if (sourceKey === "mjh-plex-us") return "plex";
  if (/amagi|wurl|xumo/i.test(name)) return "amagi";
  return "unknown";
}

/** Adjust fuzzy match score — explicit bindings (100) are unaffected. */
export function streamEpgAffinityBonus(
  streamUrl: string | undefined,
  channelName: string,
  sourceKey?: string,
): number {
  const stream = detectStreamPlatform(streamUrl);
  const channel = detectChannelPlatform(channelName, sourceKey);

  if (stream === "unknown" || channel === "unknown") return 0;

  if (stream === channel) return 18;

  if (stream === "linear" && (channel === "pluto" || channel === "roku" || channel === "samsung")) {
    return -8;
  }

  if (stream === "amagi" && (channel === "pluto" || channel === "roku" || channel === "samsung")) {
    return 6;
  }

  if (stream === "pluto" && channel === "roku") return -12;
  if (stream === "roku" && channel === "pluto") return -12;

  return 0;
}

/** US linear tvg-id feed hints for timezone correction when EPG stamps local time as UTC. */
export function preferredOffsetHoursFromTvgFeed(tvgId?: string): number[] {
  if (!tvgId?.includes("@")) return [];
  const feed = tvgId.split("@")[1]?.toLowerCase() ?? "";
  if (feed === "east" || feed.includes("eastern")) return [4, 5, -4, 0];
  if (feed === "west" || feed.includes("pacific")) return [7, 8, -7, 0];
  if (feed === "central") return [5, 6, -5, 0];
  return [];
}
