import type { EpgProgram } from "../liveMedia/epg/types";

/** Live/sports segment labels — TMDB movie search returns false matches (e.g. "Billiards"). */
const SKIP_TMDB_TITLES = new Set([
  "billiards",
  "billiard",
  "pool",
  "snooker",
  "darts",
  "news",
  "live",
  "sports",
  "sport",
  "breaking",
  "highlights",
  "replay",
  "encore",
  "paid programming",
  "to be announced",
  "tba",
]);

/** When EPG has no season/episode, skip TMDB for generic one-word live labels. */
export function shouldUseTmdbForEpg(program: EpgProgram | null | undefined): boolean {
  if (!program?.title?.trim()) return false;
  if (program.season != null || program.episode != null) return true;

  const title = program.title.trim().toLowerCase();
  if (SKIP_TMDB_TITLES.has(title)) return false;

  const words = title.split(/\s+/).filter(Boolean);
  if (words.length >= 3) return true;
  if (words.length === 2 && title.length >= 10) return true;

  // Single short token — usually a live block label, not a film/show title.
  if (words.length === 1 && title.length < 14) return false;

  return true;
}
