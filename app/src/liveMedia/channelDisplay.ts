import type { UnifiedSearchHit } from "../searchResults/types";
import { channelToSearchHit } from "./adapters";
import {
  buildChannelCardSnippet,
  resolveBroadcastLanguage,
  resolveChannelImageUrl,
  resolveChannelStreamUrl,
  resolveDisplayName,
  resolveUserCategory,
  type ChannelUserOverride,
  type UserChannelCategory,
  type ViewLanguageCode,
} from "./channelUserTaxonomy";
import { channelQualityScore, statusLabelHe } from "./ranking";
import type { Channel } from "./types";
import type { LiveMediaUserPrefs } from "./userPrefs";

export function stripLivetvId(hitId: string): string {
  return hitId.replace(/^livetv-/, "");
}

export function channelToDisplayHit(c: Channel, prefs?: LiveMediaUserPrefs | null): UnifiedSearchHit {
  const base = channelToSearchHit(c);
  if (!prefs) return base;
  return applyPrefsToHit(base, c, prefs);
}

export function applyPrefsToHit(
  hit: UnifiedSearchHit,
  c: Channel,
  prefs: LiveMediaUserPrefs,
): UnifiedSearchHit {
  const channelId = c.id;
  const overrides = prefs.channelOverrides ?? {};
  const category = resolveUserCategory(channelId, c, overrides);
  const broadcastLanguage = resolveBroadcastLanguage(c, overrides[channelId]);
  const displayName = resolveDisplayName(channelId, c, overrides);
  const imageUrl = resolveChannelImageUrl(channelId, c, overrides);
  const streamUrl = resolveChannelStreamUrl(channelId, c, overrides);
  const score = c.qualityScore ?? channelQualityScore(c);
  const snippet = buildChannelCardSnippet(c, {
    category,
    broadcastLanguage,
    he: true,
    statusLabel: statusLabelHe(c.status),
    score,
  });
  const snippetEn = buildChannelCardSnippet(c, {
    category,
    broadcastLanguage,
    he: false,
    statusLabel: statusLabelHe(c.status),
    score,
  });

  return {
    ...hit,
    title: displayName,
    titleOriginal: hit.titleOriginal || c.name,
    imageUrl: imageUrl || hit.imageUrl,
    url: streamUrl,
    mediaPlayUrl: streamUrl,
    snippet,
    snippetOriginal: snippetEn,
    meta: {
      ...hit.meta,
      channelId,
      userCategory: category,
      broadcastLanguage,
      catalogCategory: c.category,
      epgTitle: c.name,
    },
  };
}

export type ChannelDisplayMeta = {
  channelId?: string;
  userCategory?: UserChannelCategory;
  broadcastLanguage?: ViewLanguageCode;
  catalogCategory?: string;
  epgTitle?: string;
};

export function hitChannelId(hit: UnifiedSearchHit): string | null {
  const id = hit.meta?.channelId;
  if (typeof id === "string" && id) return id;
  if (hit.kind === "livetv") return stripLivetvId(hit.id);
  return null;
}
