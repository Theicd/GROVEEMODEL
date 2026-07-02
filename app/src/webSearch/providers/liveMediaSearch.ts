import type { SearchSourceResult } from "../types";
import type { LiveMediaSerpHit } from "../types";
import { buildLiveMediaCatalogSummary, ensureLiveMediaLibrary } from "../../liveMedia/catalogStore";
import { pickRegionalRadioStations } from "../../liveMedia/cableTunerRadio";
import {
  isRadioBrowseQuery,
  isRadioMediaQuery,
  resolveLiveMediaKind,
  type LiveMediaKind,
} from "../../liveMedia/mediaIntent";
import { searchLiveMediaChannels, searchLiveMediaRadio } from "../../liveMedia/search";
import { resolveCategoryFromQuery } from "../../liveMedia/queryMatch";
import type { Channel, RadioStation } from "../../liveMedia/types";
import { shouldSearchLiveMedia } from "../intents";

const MAX_TV = 32;
const MAX_RADIO = 24;

export type LiveMediaSearchOptions = {
  panelSearch?: boolean;
  /** User country for regional radio boost (ISO-2, e.g. IL). */
  countryCode?: string;
  /** Override auto radio/TV detection. */
  mediaKind?: LiveMediaKind;
  /** Skip intent gate — used when chat already routed to the local catalog. */
  catalogSearch?: boolean;
};

function channelHit(c: Channel, score: number): LiveMediaSerpHit {
  const tags = c.tags?.join(", ") || c.category;
  return {
    id: `livetv-${c.id}`,
    mediaType: "livetv",
    title: c.name,
    url: c.stream,
    streamUrl: c.stream,
    logoUrl: c.logo || undefined,
    snippet: [c.category, c.country, tags].filter(Boolean).join(" · "),
    country: c.country,
    category: c.category,
    tags: c.tags,
    status: c.status,
    fuseScore: score,
  };
}

function radioHit(r: RadioStation, score: number): LiveMediaSerpHit {
  const tags = r.tags.join(", ");
  const meta = [r.codec, r.bitrate ? `${r.bitrate}kbps` : "", tags].filter(Boolean).join(" · ");
  return {
    id: `radio-${r.id}`,
    mediaType: "radio",
    title: r.name,
    url: r.stream,
    streamUrl: r.stream,
    logoUrl: r.favicon || undefined,
    snippet: meta || r.country,
    country: r.countrycode || r.country,
    tags: r.tags,
    status: r.status,
    bitrate: r.bitrate,
    codec: r.codec,
    votes: r.votes,
    fuseScore: score,
  };
}

function emptyResult(started: number, error?: string): SearchSourceResult {
  return {
    provider: "live-tv",
    label: "TV LIVE / Radio",
    ok: !error,
    text: "",
    error,
    latencyMs: Math.round(performance.now() - started),
    liveMediaHits: [],
  };
}

function sortMergedHits(hits: LiveMediaSerpHit[]): LiveMediaSerpHit[] {
  return [...hits].sort((a, b) => {
    const rank = (s?: string) => (s === "working" ? 0 : s === "unknown" ? 1 : s === "warning" ? 2 : 3);
    const d = rank(a.status) - rank(b.status);
    if (d !== 0) return d;
    return (b.fuseScore ?? 0) - (a.fuseScore ?? 0);
  });
}

export async function fetchLiveMediaSearch(
  query: string,
  options?: LiveMediaSearchOptions,
): Promise<SearchSourceResult> {
  const started = performance.now();
  const q = query.trim();
  const catalogCategory = resolveCategoryFromQuery(q);
  if (
    !options?.catalogSearch &&
    !catalogCategory &&
    !shouldSearchLiveMedia(q, options?.panelSearch)
  ) {
    return emptyResult(started);
  }

  const countryCode = options?.countryCode?.trim().toUpperCase() || "IL";
  const mediaKind =
    options?.mediaKind ??
    (catalogCategory ? "livetv" : resolveLiveMediaKind(q));

  try {
    const { channels, radio } = await ensureLiveMediaLibrary();
    if (channels.length === 0 && radio.length === 0) {
      const summary = await buildLiveMediaCatalogSummary();
      const err = summary.lastError
        ? `קטלוג ריק — ${summary.lastError}`
        : "קטלוג TV/רדיו ריק — פתח TV LIVE מהתפריט ולחץ «סנכרון מקורות»";
      return emptyResult(started, err);
    }

    let merged: LiveMediaSerpHit[] = [];

    if (mediaKind === "radio" && isRadioBrowseQuery(q)) {
      const regional = pickRegionalRadioStations(radio, countryCode, MAX_RADIO);
      merged = regional.map((r, i) => radioHit(r, 1 - i * 0.02));
    } else {
      const tvChannels =
        mediaKind !== "radio" ? searchLiveMediaChannels(channels, q, MAX_TV) : [];
      const radioStations =
        mediaKind !== "livetv"
          ? searchLiveMediaRadio(radio, q, MAX_RADIO, countryCode)
          : [];

      const tvHits = tvChannels.map((c, i) => channelHit(c, 1 - i * 0.02));
      const radioHits = radioStations.map((r, i) => radioHit(r, 1 - i * 0.02));

      if (mediaKind === "radio") {
        merged = radioHits;
      } else if (mediaKind === "livetv") {
        merged = tvHits;
      } else {
        merged = sortMergedHits([...tvHits, ...radioHits]);
      }
    }

    const text =
      merged.length > 0
        ? merged
            .slice(0, 8)
            .map((h) => `${h.mediaType === "radio" ? "📻" : "📺"} ${h.title} — ${h.snippet}`)
            .join("\n")
        : "";

    return {
      provider: "live-tv",
      label: mediaKind === "radio" ? "Radio" : mediaKind === "livetv" ? "TV LIVE" : "TV LIVE / Radio",
      ok: true,
      text,
      latencyMs: Math.round(performance.now() - started),
      liveMediaHits: merged,
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Live media search failed";
    return emptyResult(started, msg);
  }
}

/** Re-export for chat inline routing. */
export { isRadioMediaQuery, resolveLiveMediaKind };
