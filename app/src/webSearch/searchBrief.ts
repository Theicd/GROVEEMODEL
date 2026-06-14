import type { SearchIntent, SearchProviderId, SearchSourceResult, SearchBrief, SearchBriefLink } from "./types";
import { buildDataAgeLines } from "./dataAge";
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
  const header = lines
    .filter((l) => /^(סה"|הרעידה|מסונן|לא נמצא|אין רעיד)/i.test(l.trim()))
    .map((l) => truncate(l));
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

const formatShips = (text: string): string[] => {
  const lines = text.split("\n").filter(Boolean);
  const priority = lines.filter((l) =>
    /^(אזור:|ספינות בטווח:|תשתיות ימיות|הערה:)/i.test(l.trim()),
  );
  const items = lines.filter((l) => /^\d+\./.test(l.trim())).slice(0, 6);
  const picked = [...priority, ...items].slice(0, 8);
  return picked.map((l) => truncate(l.trim()));
};

const formatMarineInfra = (text: string): string[] => {
  const lines = text.split("\n").filter(Boolean);
  const priority = lines.filter((l) =>
    /^(אזור:|תשתיות ימיות|הערה:)/i.test(l.trim()),
  );
  const items = lines.filter((l) => /^\d+\./.test(l.trim())).slice(0, 5);
  return [...priority, ...items].slice(0, 8).map((l) => truncate(l.trim()));
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
  "ais-ships": formatShips,
  "osm-overpass-marine": formatMarineInfra,
  celestrak: formatGeneric,
  "starlink-catalog": formatGeneric,
  "spacex-launches": formatGeneric,
  "hacker-news": formatGeneric,
  "adsb-aviation": formatGeneric,
  "iss-tracker": formatGeneric,
};

export const buildSearchBrief = (
  sources: SearchSourceResult[],
  intents: SearchIntent[],
  _query: string,
  _maxChars = 800,
): SearchBrief => {
  const facts: string[] = [];
  const maxFacts = intents.length >= 2 ? 14 : MAX_FACTS;
  const links: SearchBriefLink[] = [];
  const gaps: string[] = [];

  for (const s of sources) {
    if (s.ok && s.text.trim()) {
      const fmt = providerFormatters[s.provider] ?? formatGeneric;
      for (const f of fmt(s.text)) {
        if (facts.length >= maxFacts) break;
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

export const formatSearchBriefContext = (
  brief: SearchBrief,
  query: string,
  maxChars = 900,
  sources: SearchSourceResult[] = [],
): string => {
  const lines = [
    `[SEARCH BRIEF — live data for: ${truncate(query, 80)}]`,
    "Use ONLY facts below. Cite source labels. Do NOT invent numbers, names, or URLs.",
    "If ANSWER line exists — lead with it in Hebrew. If GAPS exist — mention them honestly first.",
    "If DATA AGE exists — say «עדכון אחרון מ-…», NOT «כרגע» as intraday.",
    "Max 4 sentences. No philosophy. No follow-up questions.",
  ];
  const dataAgeLines = buildDataAgeLines(sources);
  if (dataAgeLines.length) {
    lines.push(...dataAgeLines);
  }
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
  if (brief.intents.includes("ships") && brief.facts.some((f) => /ANSWER \(ships live\)|ספינות בטווח:/.test(f))) {
    const countFact =
      brief.facts.find((f) => /ANSWER \(ships live\)/.test(f)) ??
      brief.facts.find((f) => /ספינות בטווח:/.test(f));
    if (countFact) lines.splice(2, 0, `ANSWER (ships): ${countFact.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  if (brief.intents.includes("aviation") && brief.facts.some((f) => /מטוסים בטווח:/.test(f))) {
    const countFact = brief.facts.find((f) => /מטוסים בטווח:/.test(f));
    if (countFact) lines.splice(4, 0, `ANSWER (aircraft count): ${countFact.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  if (brief.intents.includes("marine-infra") && brief.facts.some((f) => /תשתיות ימיות/.test(f))) {
    const infraFact = brief.facts.find((f) => /תשתיות ימיות/.test(f));
    if (infraFact) lines.splice(4, 0, `ANSWER (marine infra): ${infraFact.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  const newsHeadline = brief.facts.find((f) => /^\[חדשות/.test(f) && /\d+\./.test(f));
  if (brief.intents.includes("news") && newsHeadline) {
    lines.splice(4, 0, `ANSWER (headline): ${newsHeadline.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  const queryWantsPm = /ראש\s+(?:ה)?ממשל(?:ה|ת)|prime\s+minister/i.test(query);
  const govHeadFact = brief.facts.find((f) => /ANSWER:\s*ראש הממשלה/i.test(f));
  const govPmFact = brief.facts.find(
    (f) =>
      (/^-\s/.test(f.trim()) || /^\[/.test(f)) &&
      /ראש ממשלה|prime minister/i.test(f) &&
      !/ראש מדינה|נשיא|head of state|monarch/i.test(f),
  );
  const govLine = queryWantsPm
    ? govHeadFact ?? govPmFact ?? brief.facts.find((f) => /ראש ממשלה/i.test(f))
    : govHeadFact ??
      brief.facts.find((f) => /ANSWER:/i.test(f)) ??
      brief.facts.find((f) => /ראש ממשלה|נשיא/i.test(f));
  if (brief.intents.includes("government") && govLine) {
    const govAnswer = govLine.includes("ANSWER:")
      ? govLine.replace(/^[^\n]*ANSWER:\s*ראש הממשלה \(Wikidata\):\s*/, "").replace(/^[^\n]*ANSWER:\s*/, "")
      : govLine.replace(/^\[[^\]]+\]\s*/, "").replace(/^-\s*/, "");
    lines.splice(4, 0, `ANSWER (government): ${govAnswer}`);
  }
  const issFact = brief.facts.find((f) => /קו רוחב:|ANSWER \(ISS position\)/i.test(f));
  if (brief.intents.includes("satellite") && issFact) {
    lines.splice(4, 0, `ANSWER (ISS): ${issFact.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  if (/\bawacs\b/i.test(query)) {
    const awacsFact = brief.facts.find((f) => /ANSWER \(AWACS\)|AWACS\?/i.test(f));
    lines.splice(
      4,
      0,
      awacsFact
        ? `ANSWER (AWACS): ${awacsFact.replace(/^\[[^\]]+\]\s*/, "").replace(/^ANSWER \(AWACS\):\s*/i, "")}`
        : "ANSWER (AWACS): השתמש ב-FACTS מעולם חי / ADS-B — ספירת AWACS heuristic.",
    );
  }
  if (brief.intents.length >= 2) {
    lines.push("CROSS-SOURCE: Compare FACTS from each source; say yes/no/partly + cite labels.");
  }
  if (brief.intents.some((i) => ["ships", "marine-infra", "aviation", "satellite", "earthquake"].includes(i))) {
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
