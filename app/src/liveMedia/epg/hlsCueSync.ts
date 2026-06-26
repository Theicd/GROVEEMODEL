/** Parsed from #EXT-X-CUE-OUT / #EXT-X-CUE-OUT-CONT in HLS media playlists. */
export type HlsCueState = {
  /** Minutes elapsed in the current programme segment (from stream, not EPG). */
  elapsedMinutes: number;
  /** Total programme segment length in minutes (stream clock). */
  durationMinutes: number;
};

const CUE_CONT_RE = /#EXT-X-CUE-OUT-CONT:ElapsedTime=([0-9.]+),Duration=([0-9.]+)/i;
const CUE_OUT_RE = /#EXT-X-CUE-OUT:([0-9.]+)/;

/** Gravitas / SSAI feeds use minutes when values exceed ~45 (not ad seconds). */
export function normalizeCueTimes(elapsed: number, duration: number): HlsCueState | null {
  if (!Number.isFinite(elapsed) || !Number.isFinite(duration) || duration <= 0) return null;

  const asMinutes = duration > 45 || elapsed > 45;
  if (asMinutes) {
    return {
      elapsedMinutes: elapsed,
      durationMinutes: duration,
    };
  }

  const elapsedMinutes = elapsed / 60;
  const durationMinutes = duration / 60;
  if (durationMinutes < 0.5) return null;
  return { elapsedMinutes, durationMinutes };
}

export function parseHlsCueFromPlaylist(text: string): HlsCueState | null {
  let lastCont: HlsCueState | null = null;
  for (const line of text.split("\n")) {
    const cont = line.match(CUE_CONT_RE);
    if (cont) {
      const parsed = normalizeCueTimes(+cont[1], +cont[2]);
      if (parsed) lastCont = parsed;
      continue;
    }
    const out = line.match(CUE_OUT_RE);
    if (out) {
      const dur = +out[1];
      const parsed = normalizeCueTimes(0, dur);
      if (parsed) lastCont = parsed;
    }
  }
  return lastCont;
}

function proxyUrl(target: string): string {
  return `/api/proxy?url=${encodeURIComponent(target)}`;
}

async function fetchPlaylistText(url: string): Promise<string | null> {
  const attempts = [url];
  if (typeof window !== "undefined") attempts.push(proxyUrl(url));

  for (const attempt of attempts) {
    try {
      const res = await fetch(attempt);
      if (!res.ok) continue;
      const text = await res.text();
      if (text.includes("#EXTM3U")) return text;
    } catch {
      /* try proxy */
    }
  }
  return null;
}

/** Pick the highest-bandwidth variant URI from a master playlist. */
export function pickVariantPlaylistUri(masterText: string, baseUrl: string): string | null {
  const lines = masterText.split("\n");
  let bestBw = 0;
  let bestUri: string | null = null;
  for (let i = 0; i < lines.length; i++) {
    const inf = lines[i].match(/#EXT-X-STREAM-INF:.*BANDWIDTH=(\d+)/i);
    if (!inf) continue;
    const uri = lines[i + 1]?.trim();
    if (!uri || uri.startsWith("#")) continue;
    const bw = +inf[1];
    if (bw >= bestBw) {
      bestBw = bw;
      bestUri = uri;
    }
  }
  if (!bestUri) return null;
  try {
    return new URL(bestUri, baseUrl).href;
  } catch {
    return null;
  }
}

/** Read live programme timing from HLS SCTE cue tags (synced to actual broadcast). */
export async function fetchHlsCueState(streamUrl: string): Promise<HlsCueState | null> {
  if (!/\.m3u8(\?|$)/i.test(streamUrl)) return null;

  const master = await fetchPlaylistText(streamUrl);
  if (!master) return null;

  let mediaUrl = streamUrl;
  if (/#EXT-X-STREAM-INF/i.test(master)) {
    const variant = pickVariantPlaylistUri(master, streamUrl);
    if (!variant) return null;
    mediaUrl = variant;
  }

  const media = await fetchPlaylistText(mediaUrl);
  if (!media) return null;
  return parseHlsCueFromPlaylist(media);
}

export function cueWindow(now: Date, cue: HlsCueState): { start: Date; end: Date } {
  const startMs = now.getTime() - cue.elapsedMinutes * 60_000;
  const endMs = startMs + cue.durationMinutes * 60_000;
  return { start: new Date(startMs), end: new Date(endMs) };
}
