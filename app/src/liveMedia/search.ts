import Fuse from "fuse.js";
import type { Channel, RadioStation, SearchFilter } from "./types";
import { expandLiveMediaSearchTerms, extractChannelDigits, isChannelNameQuery, resolveCategoryFromQuery } from "./queryMatch";
import {
  channelLanguageMatches,
  countryMatches,
  countryMatchesRadio,
  radioLanguageMatches,
  rankChannels,
  rankRadio,
} from "./ranking";

export function makeChannelIndex(channels: Channel[]) {
  return new Fuse(
    channels.filter((c) => c.type === "tv" || c.type === "youtube"),
    {
      keys: [
        { name: "name", weight: 0.45 },
        { name: "groupTitle", weight: 0.12 },
        { name: "tvgId", weight: 0.08 },
        { name: "country", weight: 0.12 },
        { name: "language", weight: 0.1 },
        { name: "category", weight: 0.13 },
        { name: "tags", weight: 0.05 },
      ],
      threshold: 0.48,
      ignoreLocation: true,
      includeScore: true,
    },
  );
}

export function makeRadioIndex(stations: RadioStation[]) {
  return new Fuse(stations, {
    keys: [
      { name: "name", weight: 0.5 },
      { name: "country", weight: 0.12 },
      { name: "countrycode", weight: 0.1 },
      { name: "language", weight: 0.13 },
      { name: "tags", weight: 0.15 },
    ],
    threshold: 0.42,
    ignoreLocation: true,
    includeScore: true,
  });
}

export function searchChannelsFuse(index: Fuse<Channel>, filter: SearchFilter): Channel[] {
  let results: Channel[];
  if (filter.query?.trim()) {
    results = index.search(filter.query).map((r) => r.item);
  } else {
    results = [];
  }
  return applyChannelFilters(results, filter);
}

export function searchRadioFuse(index: Fuse<RadioStation>, filter: SearchFilter): RadioStation[] {
  let results: RadioStation[];
  if (filter.query?.trim()) {
    results = index.search(filter.query).map((r) => r.item);
  } else {
    results = [];
  }
  return applyRadioFilters(results, filter);
}

export function applyChannelFilters(channels: Channel[], filter: SearchFilter): Channel[] {
  const geoFilter = !!(filter.country || filter.language);
  return channels.filter((c) => {
    if (filter.country && !countryMatches(c, filter.country)) return false;
    if (filter.language && !channelLanguageMatches(c, filter.language)) return false;
    if (filter.category && c.category !== filter.category) return false;
    if (filter.onlyFavorites && !c.favorite) return false;
    if (filter.onlyWorking && c.status !== "working") return false;
    // When browsing by country/language, keep offline visible (sorted last) so lists aren't empty.
    if (!geoFilter && c.status === "offline") return false;
    return true;
  });
}

export function applyRadioFilters(stations: RadioStation[], filter: SearchFilter): RadioStation[] {
  const geoFilter = !!filter.country;
  return stations.filter((s) => {
    if (filter.country && !countryMatchesRadio(s, filter.country)) return false;
    if (filter.language && !radioLanguageMatches(s, filter.language)) return false;
    if (filter.onlyFavorites && !s.favorite) return false;
    if (filter.onlyWorking && s.status !== "working") return false;
    if (!geoFilter && s.status === "offline") return false;
    return true;
  });
}

/** Browse/sort TV list for panel — filters + quality ranking. */
export function listTvChannelsForPanel(
  channels: Channel[],
  opts: {
    category?: string;
    country?: string;
    language?: string;
    query?: string;
    limit?: number;
  },
): Channel[] {
  let list = channels.filter((c) => c.type === "tv" || c.type === "youtube");
  list = applyChannelFilters(list, {
    query: "",
    category: opts.category || undefined,
    country: opts.country || undefined,
    language: opts.language || undefined,
  });
  if (opts.query?.trim()) {
    list = searchLiveMediaChannels(list, opts.query.trim(), opts.limit ?? 500);
  }
  list = rankChannels(list);
  if (opts.limit) list = list.slice(0, opts.limit);
  return list;
}

/** Browse/sort radio list for panel. */
export function listRadioForPanel(
  stations: RadioStation[],
  opts: { country?: string; language?: string; query?: string; limit?: number },
): RadioStation[] {
  let list = applyRadioFilters(stations, {
    query: "",
    country: opts.country || undefined,
    language: opts.language || undefined,
  });
  if (opts.query?.trim()) {
    list = searchLiveMediaRadio(list, opts.query.trim(), opts.limit ?? 500);
  }
  list = rankRadio(list);
  if (opts.limit) list = list.slice(0, opts.limit);
  return list;
}

function visibleChannels(channels: Channel[], showOffline = false): Channel[] {
  return channels.filter(
    (c) =>
      (c.type === "tv" || c.type === "youtube") &&
      (showOffline || c.status !== "offline"),
  );
}

/** Search TV channels — category aliases, fuse, and channel-number heuristics. */
export function searchLiveMediaChannels(channels: Channel[], query: string, limit = 24): Channel[] {
  const q = query.trim();
  if (!q) return [];
  const pool = visibleChannels(channels);
  const category = resolveCategoryFromQuery(q);
  if (category) {
    const byCat = pool.filter((c) => c.category === category);
    if (byCat.length) return byCat.slice(0, limit);
  }
  const digits = extractChannelDigits(q);
  if (digits || isChannelNameQuery(q)) {
    const byNum = pool.filter((c) => {
      const name = c.name.toLowerCase();
      if (digits && name.includes(digits)) return true;
      return q.split(/\s+/).every((w) => w.length >= 2 && name.includes(w.toLowerCase()));
    });
    if (byNum.length) return byNum.slice(0, limit);
  }
  const index = makeChannelIndex(pool);
  const terms = expandLiveMediaSearchTerms(q);
  const seen = new Map<string, Channel>();
  for (const term of terms) {
    for (const r of index.search(term)) {
      if (!seen.has(r.item.id)) seen.set(r.item.id, r.item);
    }
  }
  if (seen.size) return [...seen.values()].slice(0, limit);
  return index.search(q).map((r) => r.item).slice(0, limit);
}

/** Search radio stations with fuse + tag/name substring fallback. */
export function searchLiveMediaRadio(stations: RadioStation[], query: string, limit = 18): RadioStation[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const pool = stations.filter((s) => s.status !== "offline");
  const index = makeRadioIndex(pool);
  const fuseResults = index.search(q).map((r) => r.item);
  if (fuseResults.length) return fuseResults.slice(0, limit);
  return pool
    .filter(
      (s) =>
        s.name.toLowerCase().includes(q) ||
        s.tags.some((t) => t.toLowerCase().includes(q)) ||
        s.country.toLowerCase().includes(q),
    )
    .slice(0, limit);
}
