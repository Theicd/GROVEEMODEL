import type { Channel, RadioStation, StreamStatus } from "./types";
const STATUS_RANK: Record<StreamStatus, number> = {
  working: 0,
  warning: 1,
  unknown: 2,
  offline: 3,
};

/** ISO 639-1 / 639-3 aliases used in IPTV feeds. */
const LANG_ALIASES: Record<string, string[]> = {
  heb: ["heb", "he", "hebrew", "עברית"],
  eng: ["eng", "en", "english", "אנגלית"],
  ara: ["ara", "ar", "arabic", "ערבית"],
  rus: ["rus", "ru", "russian", "רוסית"],
  fra: ["fra", "fr", "french", "צרפתית"],
  deu: ["deu", "de", "german", "גרמנית"],
  hin: ["hin", "hi", "hindi", "הינדי", "bhojpuri", "bollywood"],
  urd: ["urd", "ur", "urdu", "אורדו"],
  spa: ["spa", "es", "spanish", "ספרדית"],
  por: ["por", "pt", "portuguese"],
  ita: ["ita", "it", "italian"],
  tur: ["tur", "tr", "turkish"],
  zho: ["zho", "zh", "chinese"],
  jpn: ["jpn", "ja", "japanese"],
  kor: ["kor", "ko", "korean"],
  tam: ["tam", "ta", "tamil"],
  tel: ["tel", "te", "telugu"],
  ben: ["ben", "bn", "bengali"],
};

function normCode(v: string): string {
  return v.trim().toLowerCase();
}

export function languageMatches(channelLang: string | undefined, filterCode: string): boolean {
  if (!filterCode) return true;
  const raw = normCode(channelLang || "");
  const aliases = LANG_ALIASES[filterCode] ?? [filterCode];
  if (!raw) return false;
  const parts = raw.split(/[,;|/\s]+/).filter(Boolean);
  for (const alias of aliases) {
    const a = normCode(alias);
    if (raw === a || raw.startsWith(`${a}-`)) return true;
    if (parts.some((p) => p === a || p.startsWith(`${a}-`))) return true;
  }
  return false;
}

export function channelLanguageMatches(channel: Channel, filterCode: string): boolean {
  if (!filterCode) return true;
  const langs = channel.languages?.length ? channel.languages : channel.language ? [channel.language] : [];
  if (!langs.length) return false;
  return langs.some((l) => languageMatches(l, filterCode));
}

export function radioLanguageMatches(station: RadioStation, filterCode: string): boolean {
  if (!filterCode) return true;
  const langs = station.languages?.length ? station.languages : station.language ? [station.language] : [];
  if (!langs.length) return false;
  return langs.some((l) => languageMatches(l, filterCode));
}

export function countryMatches(channel: Channel, filterCode: string): boolean {
  if (!filterCode) return true;
  const code = normCode(filterCode);
  const cCountry = normCode(channel.country || "");
  if (cCountry === code) return true;
  if (channel.source === `iptv-org-${code}`) return true;
  if (channel.source.includes(code) && channel.source.includes("iptv-org")) return true;
  if (channel.tags?.some((t) => normCode(t) === code)) return true;
  if (channel.groupTitle?.toLowerCase().includes(code)) return true;
  return false;
}

export function countryMatchesRadio(station: RadioStation, filterCode: string): boolean {
  if (!filterCode) return true;
  const code = normCode(filterCode);
  return normCode(station.countrycode || "") === code;
}

export function channelQualityScore(channel: Channel): number {
  let score = 0;
  if (channel.status === "working") score += 100;
  else if (channel.status === "warning") score += 65;
  else if (channel.status === "unknown") score += 45;
  else score += 5;

  if (channel.loadTimeMs != null) {
    if (channel.loadTimeMs <= 1500) score += 25;
    else if (channel.loadTimeMs <= 4000) score += 15;
    else if (channel.loadTimeMs <= 9000) score += 8;
    else score += 2;
  } else {
    score += 10;
  }

  if (channel.logo) score += 4;
  if (channel.country) score += 3;
  if (channel.language) score += 2;
  if (channel.favorite) score += 6;
  return score;
}

export function radioQualityScore(station: RadioStation): number {
  let score = 0;
  if (station.status === "working") score += 100;
  else if (station.status === "warning") score += 65;
  else if (station.status === "unknown") score += 45;
  else score += 5;
  if (station.loadTimeMs != null) {
    if (station.loadTimeMs <= 1500) score += 20;
    else if (station.loadTimeMs <= 4000) score += 12;
    else score += 4;
  }
  if (station.votes) score += Math.min(15, Math.round(Math.log10(station.votes + 1) * 5));
  if (station.bitrate && station.bitrate >= 128) score += 4;
  if (station.favorite) score += 6;
  return score;
}

export function rankChannels(channels: Channel[]): Channel[] {
  return [...channels].sort((a, b) => {
    const sr = STATUS_RANK[a.status] - STATUS_RANK[b.status];
    if (sr !== 0) return sr;
    const qs = channelQualityScore(b) - channelQualityScore(a);
    if (qs !== 0) return qs;
    const lt =
      (a.loadTimeMs ?? Number.MAX_SAFE_INTEGER) - (b.loadTimeMs ?? Number.MAX_SAFE_INTEGER);
    if (lt !== 0) return lt;
    return a.name.localeCompare(b.name, "he");
  });
}

export function rankRadio(stations: RadioStation[]): RadioStation[] {
  return [...stations].sort((a, b) => {
    const sa = a.status ?? "unknown";
    const sb = b.status ?? "unknown";
    const sr = STATUS_RANK[sa] - STATUS_RANK[sb];
    if (sr !== 0) return sr;
    return radioQualityScore(b) - radioQualityScore(a) || a.name.localeCompare(b.name, "he");
  });
}

export function statusLabelHe(status: StreamStatus | undefined): string {
  switch (status) {
    case "working":
      return "פעיל";
    case "warning":
      return "איטי";
    case "offline":
      return "לא פעיל";
    default:
      return "לא נבדק";
  }
}
