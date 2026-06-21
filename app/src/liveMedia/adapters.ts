import type { Channel, RadioStation } from "./types";
import type { UnifiedSearchHit } from "../searchResults/types";
import { channelQualityScore, radioQualityScore, statusLabelHe } from "./ranking";
import { LIVE_MEDIA_COUNTRIES } from "./catalogs";
import { languageDisplayLabel } from "./languageMetadata";

function countryLabel(code: string, he: boolean): string {
  const c = LIVE_MEDIA_COUNTRIES.find((x) => x.code === code.toLowerCase());
  if (!c) return code.toUpperCase();
  return he ? c.nameHe : c.name;
}

function languagesLabel(c: Channel | RadioStation, he = true): string {
  const codes = c.languages?.length ? c.languages : c.language ? [c.language] : [];
  if (!codes.length) return he ? "שפה לא ידועה" : "Unknown lang";
  return codes.map((code) => languageDisplayLabel(code, he)).join(", ");
}

function channelSnippet(c: Channel, he = true): string {
  const parts: string[] = [];
  if (c.country) parts.push(countryLabel(c.country, he));
  parts.push(languagesLabel(c, he));
  if (c.category) parts.push(c.category);
  if (c.languageSource === "inferred") parts.push(he ? "שפה מזוהה" : "lang detected");
  parts.push(statusLabelHe(c.status));
  if (c.loadTimeMs != null && c.loadTimeMs > 0) {
    parts.push(he ? `טעינה ${(c.loadTimeMs / 1000).toFixed(1)}ש׳` : `load ${(c.loadTimeMs / 1000).toFixed(1)}s`);
  }
  const score = c.qualityScore ?? channelQualityScore(c);
  parts.push(he ? `ניקוד ${score}` : `score ${score}`);
  return parts.join(" · ");
}

export function channelToSearchHit(c: Channel, fuseScore = 0.5): UnifiedSearchHit {
  const tags = c.tags?.join(", ") || c.category;
  const score = c.qualityScore ?? channelQualityScore(c);
  return {
    id: `livetv-${c.id}`,
    kind: "livetv",
    title: c.name,
    titleOriginal: c.name,
    url: c.stream,
    snippet: channelSnippet(c, true) || [c.category, tags].filter(Boolean).join(" · "),
    snippetOriginal: channelSnippet(c, false) || [c.category, tags].filter(Boolean).join(" · "),
    imageUrl: c.logo || undefined,
    mediaPlayUrl: c.stream,
    sourceLabel: "TV LIVE",
    provider: "live-tv",
    score: score + Math.round(fuseScore * 12),
    meta: {
      engine: c.category || "Live",
      year: score,
      status: c.status,
      loadTimeMs: c.loadTimeMs,
      languages: c.languages?.join(", ") || c.language,
      languageSource: c.languageSource,
    },
    summarizable: false,
  };
}

export function radioToSearchHit(r: RadioStation, fuseScore = 0.5): UnifiedSearchHit {
  const meta = [r.codec, r.bitrate ? `${r.bitrate}kbps` : "", r.tags.join(", ")].filter(Boolean).join(" · ");
  const score = r.qualityScore ?? radioQualityScore(r);
  const status = r.status ?? "unknown";
  const snippetParts = [
    r.country || r.countrycode?.toUpperCase(),
    languagesLabel(r, true),
    statusLabelHe(status),
    r.loadTimeMs ? `טעינה ${(r.loadTimeMs / 1000).toFixed(1)}ש׳` : "",
    `ניקוד ${score}`,
    meta,
  ].filter(Boolean);
  return {
    id: `radio-${r.id}`,
    kind: "radio",
    title: r.name,
    titleOriginal: r.name,
    url: r.stream,
    snippet: snippetParts.join(" · "),
    snippetOriginal: snippetParts.join(" · "),
    imageUrl: r.favicon || undefined,
    mediaPlayUrl: r.stream,
    sourceLabel: "Radio",
    provider: "live-tv",
    score: score + Math.round(fuseScore * 12),
    meta: { engine: r.codec || "Radio", year: score, status, loadTimeMs: r.loadTimeMs },
    summarizable: false,
  };
}
