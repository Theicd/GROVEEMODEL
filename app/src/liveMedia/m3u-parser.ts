import type { Channel } from "./types";

interface ParseOptions {
  source: string;
  defaultCountry?: string;
  defaultLanguage?: string;
}

function extractAttr(line: string, attr: string): string {
  const m = line.match(new RegExp(`${attr}="([^"]*)"`, "i"));
  return m ? m[1] : "";
}

function hashId(str: string): string {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h << 5) - h + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h).toString(36);
}

function guessCountry(group: string): string {
  if (!group) return "";
  for (const p of group.split(";")) {
    const t = p.trim().toLowerCase();
    if (t.length === 2) return t;
  }
  return "";
}

function guessCategory(group: string): string {
  if (!group) return "general";
  const parts = group.split(";").map((p) => p.trim().toLowerCase());
  const categories = [
    "news", "sports", "movies", "kids", "music", "anime", "documentary",
    "education", "entertainment", "business", "cooking", "religious",
    "science", "travel", "auto", "legislative", "general", "family",
    "outdoor", "relax", "series", "culture", "shop", "weather", "comedy",
  ];
  for (const p of parts) {
    if (categories.includes(p)) return p;
  }
  return parts[0] || "general";
}

export function parseM3U(text: string, opts: ParseOptions): Channel[] {
  const lines = text.split(/\r?\n/);
  const channels: Channel[] = [];
  let current: Partial<Channel> | null = null;
  let channelNumber = 1;

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;

    if (line.startsWith("#EXTINF")) {
      const nameMatch = line.match(/,(.+)$/);
      const name = nameMatch ? nameMatch[1].trim() : "";
      const groupTitle = extractAttr(line, "group-title");
      current = {
        name: name || extractAttr(line, "tvg-name"),
        logo: extractAttr(line, "tvg-logo"),
        tvgId: extractAttr(line, "tvg-id"),
        groupTitle,
        country: extractAttr(line, "tvg-country")?.toLowerCase() || guessCountry(groupTitle) || opts.defaultCountry || "",
        language: extractAttr(line, "tvg-language")?.toLowerCase().split(";")[0] || opts.defaultLanguage || "",
        category: guessCategory(groupTitle),
        tags: groupTitle ? groupTitle.split(";").map((s) => s.trim()).filter(Boolean) : [],
      };
    } else if (!line.startsWith("#") && current) {
      const stream = line;
      const name = current.name || "Unknown";
      channels.push({
        id: hashId(`${opts.source}|${stream}|${name}`),
        name,
        logo: current.logo || "",
        country: current.country || "",
        language: current.language || "",
        category: current.category || "general",
        stream,
        source: opts.source,
        type: "tv",
        status: "unknown",
        lastCheck: 0,
        favorite: false,
        tags: current.tags || [],
        tvgId: current.tvgId,
        groupTitle: current.groupTitle,
        channelNumber: channelNumber++,
        addedAt: Date.now(),
      });
      current = null;
    }
  }
  return channels;
}
