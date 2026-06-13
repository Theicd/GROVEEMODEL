import type { SearchIntent, SearchProviderId, SearchSourceResult, SearchBrief, SearchBriefLink } from "./types";
import { LIVE_WORLD_LAYERS_HE } from "./searchProviders";

export type { SearchBrief, SearchBriefLink };

const MAX_FACTS = 8;
const MAX_LINKS = 6;
const MAX_FACT_LEN = 120;

const truncate = (s: string, max = MAX_FACT_LEN) =>
  s.length <= max ? s : `${s.slice(0, max - 1).trim()}…`;

const formatGithub = (text: string): string[] =>
  text
    .split("\n")
    .filter((l) => l.startsWith("- "))
    .slice(0, 5)
    .map((l) => truncate(l.replace(/^- /, "")));

const formatWeather = (text: string): string[] => {
  const lines = text.split("\n").filter(Boolean);
  const priority = lines.filter((l) =>
    /^(מיקום|זמן|מצב|טמפר|לחות|רוח|לחץ|תחזית|גשם)/i.test(l.trim()),
  );
  const picked = (priority.length ? priority : lines).slice(0, 7);
  return picked.map((l) => truncate(l));
};

const formatEarthquake = (text: string): string[] => {
  const lines = text.split("\n").filter(Boolean);
  const header = lines[0] ? [truncate(lines[0])] : [];
  const quakes = lines.filter((l) => l.startsWith("- M")).slice(0, 6).map((l) => truncate(l));
  return [...header, ...quakes];
};

const formatWorldTime = (text: string): string[] =>
  text
    .split("\n")
    .filter(Boolean)
    .slice(0, 5)
    .map((l) => truncate(l));

const formatWikipedia = (text: string): string[] => {
  const firstBlock = text.split("\n\n")[0] ?? text;
  const sentences = firstBlock.replace(/\n/g, " ").split(/(?<=[.!?])\s+/).slice(0, 3);
  return sentences.map((s) => truncate(s.trim())).filter(Boolean);
};

const formatGeneric = (text: string): string[] =>
  text
    .split("\n")
    .filter((l) => l.trim())
    .slice(0, 4)
    .map((l) => truncate(l.trim()));

const providerFormatters: Partial<Record<SearchProviderId, (text: string) => string[]>> = {
  github: formatGithub,
  "open-meteo": formatWeather,
  "open-meteo-marine": formatGeneric,
  "world-time": formatWorldTime,
  "wikipedia-en": formatWikipedia,
  "wikipedia-he": formatWikipedia,
  "frankfurter-fx": formatGeneric,
  "rest-countries": formatGeneric,
  "usgs-earthquake": formatEarthquake,
  "nominatim-places": formatGeneric,
  "huggingface-models": formatGeneric,
  "huggingface-datasets": formatGeneric,
  "ais-ships": formatGeneric,
  celestrak: formatGeneric,
  "spacex-launches": formatGeneric,
  "hacker-news": formatGeneric,
  "adsb-aviation": formatGeneric,
};

export const buildSearchBrief = (
  sources: SearchSourceResult[],
  intents: SearchIntent[],
  _query: string,
  _maxChars = 800,
): SearchBrief => {
  const facts: string[] = [];
  const links: SearchBriefLink[] = [];
  const gaps: string[] = [];

  for (const s of sources) {
    if (s.ok && s.text.trim()) {
      const fmt = providerFormatters[s.provider] ?? formatGeneric;
      for (const f of fmt(s.text)) {
        if (facts.length >= MAX_FACTS) break;
        facts.push(`[${s.label}] ${f}`);
      }
      if (s.url && links.length < MAX_LINKS) {
        links.push({ label: s.label, url: s.url });
      }
    } else if (!s.ok && s.error) {
      gaps.push(`${s.label}: ${s.error}`);
    }
  }

  const okCount = sources.filter((s) => s.ok && s.text.trim()).length;
  if (!okCount) {
    gaps.unshift("לא נמצאו נתונים חיים לשאלה זו");
  }

  return { facts, links, gaps, intents };
};

export const formatSearchBriefContext = (brief: SearchBrief, query: string, maxChars = 800): string => {
  const lines = [
    `[SEARCH BRIEF — live data for: ${truncate(query, 80)}]`,
    "Use ONLY facts below. Cite source labels. Do NOT invent numbers or URLs.",
  ];
  if (brief.facts.length) {
    lines.push("FACTS:");
    lines.push(...brief.facts.map((f) => `- ${f}`));
  }
  if (brief.links.length) {
    lines.push("LINKS:");
    lines.push(...brief.links.map((l) => `- ${l.label}: ${l.url}`));
  }
  if (brief.gaps.length) {
    lines.push("GAPS (tell user honestly):");
    lines.push(...brief.gaps.map((g) => `- ${g}`));
  }
  if (brief.intents.some((i) => ["ships", "aviation", "satellite", "earthquake"].includes(i))) {
    lines.push(`LIVE WORLD: ${LIVE_WORLD_LAYERS_HE}`);
  }
  lines.push(`intents: ${brief.intents.join(", ")}`);
  lines.push("[/SEARCH BRIEF]");

  let out = lines.join("\n");
  if (out.length > maxChars) {
    out = `${out.slice(0, maxChars - 16).trim()}\n…[/SEARCH BRIEF]`;
  }
  return out;
};
