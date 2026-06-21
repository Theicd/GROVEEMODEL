import type { Channel, RadioStation } from "./types";

/** Name / tag patterns for channels we hide by default (religious, regional clutter). */
const DEFAULT_NAME_PATTERNS: RegExp[] = [
  /\bquran\b/i,
  /\bislam(ic)?\b/i,
  /\bmuslim\b/i,
  /\bshia\b/i,
  /\bsunni\b/i,
  /\bprayer\b/i,
  /\bsermon\b/i,
  /\bchurch\b/i,
  /\bgospel\b/i,
  /\bchristian\b/i,
  /\bjesus\b/i,
  /\bbible\b/i,
  /\bhindu\b/i,
  /\bbhojpuri\b/i,
  /\bbollywood\b/i,
  /\bb4u\b/i,
  /\bzeetv\b/i,
  /\bstar\s*plus\b/i,
  /\bsony\s*(tv|max)\b/i,
  /\burdu\b/i,
  /\btamil\b/i,
  /\btelugu\b/i,
  /\bbengali\b/i,
  /\bpunjabi\b/i,
  /\bmarathi\b/i,
  /\bmalayalam\b/i,
  /\bkannada\b/i,
  /\bgujarati\b/i,
  /\btelefe\b/i,
  /\bcaracol\b/i,
  /\btelevisa\b/i,
  /\bntv\s*(pk|bd|in)\b/i,
  /\bgeo\s*tv\b/i,
  /\bary\s*(digital|news|zindagi)\b/i,
];

const DEFAULT_CATEGORIES = new Set(["religious"]);

export function matchesDefaultBlacklistChannel(c: Channel): boolean {
  if (DEFAULT_CATEGORIES.has(c.category)) return true;
  const hay = [c.name, c.groupTitle, ...(c.tags ?? [])].filter(Boolean).join(" ");
  return DEFAULT_NAME_PATTERNS.some((re) => re.test(hay));
}

export function matchesDefaultBlacklistRadio(r: RadioStation): boolean {
  const hay = [r.name, ...r.tags].join(" ");
  return DEFAULT_NAME_PATTERNS.some((re) => re.test(hay));
}

export function collectDefaultBlacklistIds(channels: Channel[], radio: RadioStation[]): {
  channelIds: string[];
  radioIds: string[];
} {
  return {
    channelIds: channels.filter(matchesDefaultBlacklistChannel).map((c) => c.id),
    radioIds: radio.filter(matchesDefaultBlacklistRadio).map((r) => r.id),
  };
}
