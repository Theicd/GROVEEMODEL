import type { UnifiedSearchHit } from "./searchResults/types";
import type { LiveMediaSerpHit } from "./webSearch/types";
import { resolveLiveMediaKind, isSportsLiveMediaRequest } from "./liveMedia/mediaIntent";

export { isSportsLiveMediaRequest };

type UiLang = "he" | "en";

export type InlineLiveMediaPayload = {
  hits: UnifiedSearchHit[];
  mode: "livetv" | "radio";
  sportsPackage?: boolean;
};

export function liveMediaSerpHitsToUnified(hits: LiveMediaSerpHit[]): UnifiedSearchHit[] {
  return hits.map((m) => {
    const kind = m.mediaType === "radio" ? "radio" : "livetv";
    const statusBoost = m.status === "working" ? 24 : m.status === "unknown" ? 6 : 0;
    const fuseBoost = Math.round((m.fuseScore ?? 0) * 40);
    return {
      id: m.id,
      kind,
      title: m.title,
      titleOriginal: m.title,
      url: m.url,
      snippet: m.snippet?.trim() || m.category || "",
      snippetOriginal: m.snippet || "",
      imageUrl: m.logoUrl,
      mediaPlayUrl: m.streamUrl,
      sourceLabel: kind === "radio" ? "Radio" : "TV LIVE",
      provider: "live-tv",
      score: (kind === "radio" ? 68 : 58) + statusBoost + fuseBoost,
      meta: {
        engine: m.category || m.codec || "Live",
        year: m.bitrate,
      },
      summarizable: false,
    };
  });
}

export function buildSimpleGreetingReply(uiLang: UiLang): string {
  return uiLang === "he"
    ? "שלום! אני גרובי. במה אוכל לעזור?"
    : "Hi! I'm Groovie. How can I help?";
}

export function buildLiveMediaInlineReply(
  count: number,
  uiLang: UiLang,
  opts?: { sportsPackage?: boolean; radio?: boolean; movies?: boolean },
): string {
  const sportsPackage = opts?.sportsPackage;
  const radio = opts?.radio;
  const movies = opts?.movies;
  if (sportsPackage) {
    return uiLang === "he"
      ? `הנה ${count} ערוצי ספורט לדוגמה — לחץ ▶ לצפייה ישירות כאן בצ'אט. לחבילת הספורט המלאה לחץ «חבילת ספורט» למטה.`
      : `Here are ${count} sample sports channels — tap ▶ to watch inline. Tap «Sports package» below for the full lineup.`;
  }
  if (radio) {
    return uiLang === "he"
      ? `נמצאו ${count} תחנות רדיו — לחץ ▶ להאזנה ישירות כאן בצ'אט.`
      : `Found ${count} radio stations — tap ▶ to listen inline in chat.`;
  }
  if (movies) {
    return uiLang === "he"
      ? `נמצאו ${count} ערוצי סרטים — לחץ ▶ לצפייה ישירות כאן בצ'אט.`
      : `Found ${count} movie channels — tap ▶ to watch inline in chat.`;
  }
  return uiLang === "he"
    ? `נמצאו ${count} ערוצים — לחץ ▶ לצפייה ישירות כאן בצ'אט.`
    : `Found ${count} channels — tap ▶ to watch inline in chat.`;
}

export function resolveLiveMediaModeFromQuery(
  query: string,
  hits: UnifiedSearchHit[],
): "livetv" | "radio" {
  const kind = resolveLiveMediaKind(query, false);
  if (kind === "radio") return "radio";
  if (kind === "livetv") return "livetv";
  const tv = hits.filter((h) => h.kind === "livetv").length;
  const radio = hits.filter((h) => h.kind === "radio").length;
  return radio >= tv ? "radio" : "livetv";
}

export function filterUnifiedLiveMediaHits(
  hits: UnifiedSearchHit[],
  mode: "livetv" | "radio",
): UnifiedSearchHit[] {
  const filtered = hits.filter((h) => h.kind === mode);
  return filtered.length ? filtered : hits;
}
