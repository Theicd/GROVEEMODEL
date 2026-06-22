import type {
  AnswerShape,
  SearchIntent,
  SearchProviderId,
  SearchSourceResult,
  SearchBrief,
  SearchBriefLink,
} from "./types";
import { buildDataAgeLines } from "./dataAge";
import { LIVE_WORLD_LAYERS_HE } from "./searchProviders";
import { buildCrossSourceCorrelationLines,
  extractCrossSourceMetrics,
  shouldBuildCrossSourceCorrelation,
} from "./crossSourceCorrelation";
import { needsOpenWebEnrichment } from "./openWebTopics";
import {
  extractCinemaMoviesFromSources,
  parseCinemaMoviesFromText,
} from "./cinemaIlExtract";
import { isIsraelCinemaNowQuery } from "./openWebTopicDetect";

export type { SearchBrief, SearchBriefLink };

const MAX_FACTS = 8;
const MAX_LINKS = 6;
const MAX_FACT_LEN = 120;
const CINEMA_LISTING_FACT_MAX = 320;

/** Per-provider fact caps before rerank — keeps brief focused. */
const PROVIDER_FACT_CAPS: Partial<Record<SearchProviderId, number>> = {
  github: 3,
  "open-meteo": 6,
  "open-meteo-marine": 4,
  "open-meteo-air-quality": 4,
  "usgs-earthquake": 6,
  "adsb-aviation": 5,
  "ais-ships": 6,
  "osm-overpass-marine": 5,
  "wikipedia-en": 3,
  "wikipedia-he": 3,
  "grovee-news": 8,
  arxiv: 4,
  "url-context": 6,
  searxng: 4,
  tavily: 5,
  scavio: 5,
  "world-time": 4,
};

const INTENT_PROVIDER_PRIORITY: Partial<Record<SearchIntent, SearchProviderId[]>> = {
  weather: ["open-meteo"],
  airquality: ["open-meteo-air-quality", "open-meteo"],
  aviation: ["adsb-aviation"],
  ships: ["ais-ships"],
  "marine-infra": ["osm-overpass-marine", "ais-ships"],
  marine: ["open-meteo-marine"],
  earthquake: ["usgs-earthquake"],
  news: ["grovee-news"],
  government: ["wikidata-gov"],
  wikipedia: ["wikipedia-he", "wikipedia-en"],
  github: ["github"],
  link: ["url-context"],
  arxiv: ["arxiv"],
  satellite: ["iss-tracker", "celestrak"],
};

const truncate = (s: string, max = MAX_FACT_LEN) =>
  s.length <= max ? s : `${s.slice(0, max - 1).trim()}…`;

const formatGithub = (text: string): string[] =>
  text
    .split("\n")
    .filter((l) => /^(?:- |\d+\. )/.test(l.trim()))
    .slice(0, 5)
    .map((l) => truncate(l.replace(/^(?:- |\d+\.\s*)/, "")));

const formatWeather = (text: string): string[] => {
  const lines = text.split("\n").filter(Boolean);
  const priority = lines.filter((l) =>
    /^(מיקום|זמן|מצב|טמפר|לחות|רוח|לחץ|תחזית|גשם)/i.test(l.trim()),
  );
  const picked = (priority.length ? priority : lines).slice(0, 7);
  return picked.map((l) => truncate(l));
};

const formatAirQuality = (text: string): string[] => {
  const lines = text.split("\n").filter(Boolean);
  const priority = lines.filter((l) =>
    /^(מיקום|זמן|US AQI|PM2|PM10|NO₂|O₃|ANSWER \(air quality\))/i.test(l.trim()),
  );
  return (priority.length ? priority : lines).slice(0, 6).map((l) => truncate(l));
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
    /^(אזור:|ANSWER \(ships live\)|דיווח AIS|ספינות בטווח:|סימוני מסלול|הערה:|עודכן:)/i.test(l.trim()),
  );
  const items = lines.filter((l) => /^\d+\./.test(l.trim())).slice(0, 4);
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
  "open-meteo-air-quality": formatAirQuality,
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
  arxiv: formatGeneric,
};

const factProvider = (fact: string): SearchProviderId | null => {
  const labelMatch = fact.match(/^\[([^\]]+)\]/);
  if (!labelMatch) return null;
  const label = labelMatch[1].toLowerCase();
  if (/מזג|weather|open-meteo/i.test(label) && !/air|אוויר/i.test(label)) return "open-meteo";
  if (/אוויר|air quality/i.test(label)) return "open-meteo-air-quality";
  if (/ads-b|תעופה|aviation/i.test(label)) return "adsb-aviation";
  if (/ספינ|ais|ship/i.test(label)) return "ais-ships";
  if (/github/i.test(label)) return "github";
  if (/usgs|רעיד/i.test(label)) return "usgs-earthquake";
  if (/wikipedia/i.test(label)) return label.includes("he") ? "wikipedia-he" : "wikipedia-en";
  if (/wikidata|ממשל/i.test(label)) return "wikidata-gov";
  if (/חדשות|news|grovee/i.test(label)) return "grovee-news";
  if (/tavily/i.test(label)) return "tavily";
  if (/scavio/i.test(label)) return "scavio";
  if (/searxng|web search|חיפוש/i.test(label)) return "searxng";
  return null;
};

const scoreFact = (
  fact: string,
  intents: SearchIntent[],
  query: string,
  answerShape?: AnswerShape,
): number => {
  let score = 0;
  const provider = factProvider(fact);
  for (let i = 0; i < intents.length; i++) {
    const intent = intents[i];
    const prefs = INTENT_PROVIDER_PRIORITY[intent] ?? [];
    const idx = provider ? prefs.indexOf(provider) : -1;
    if (idx >= 0) score += 100 - idx * 10 - i * 5;
  }
  if (/ANSWER/i.test(fact)) score += 50;
  if (answerShape === "count" && /\d+/.test(fact)) score += 30;
  if (answerShape === "short_fact" && /ANSWER|מיקום|US AQI|מטוסים בטווח|ספינות בטווח/i.test(fact)) {
    score += 20;
  }
  if (answerShape === "overview" && provider?.startsWith("wikipedia")) score += 25;
  if (answerShape === "bullet_list") score += 5;
  if (/רוח|wind/i.test(query) && /רוח|wind/i.test(fact)) score += 40;
  if (/pm2|aqi|איכות/i.test(query) && /AQI|PM2/i.test(fact)) score += 40;
  if (intents.includes("news") && /חדשות|news|grovee/i.test(fact)) score += 40;
  if (
    needsOpenWebEnrichment(query) &&
    (provider === "tavily" || provider === "searxng" || provider === "scavio")
  ) {
    score += 150;
  }
  if (isIsraelCinemaNowQuery(query) && /ANSWER \(now showing\)|סרט בקולנוע:/i.test(fact)) {
    score += 300;
  }
  if (isIsraelCinemaNowQuery(query) && /edb\.co\.il|סרטים ישראלים/i.test(fact)) {
    score -= 80;
  }
  if (isIsraelCinemaNowQuery(query) && /HOT CINEMA רשת|עמוד הבית|יום הקולנוע הישראלי/i.test(fact)) {
    score -= 120;
  }
  return score;
};

export const rerankBriefFacts = (
  facts: string[],
  intents: SearchIntent[],
  query: string,
  answerShape?: AnswerShape,
): string[] => {
  if (facts.length <= 1) return facts;
  return [...facts].sort(
    (a, b) => scoreFact(b, intents, query, answerShape) - scoreFact(a, intents, query, answerShape),
  );
};

export const buildSearchBrief = (
  sources: SearchSourceResult[],
  intents: SearchIntent[],
  query: string,
  _maxChars = 800,
  answerShape?: AnswerShape,
): SearchBrief => {
  const facts: string[] = [];
  const maxFacts = isIsraelCinemaNowQuery(query)
    ? 22
    : intents.includes("news")
    ? 24
    : answerShape === "short_fact"
      ? 5
      : answerShape === "bullet_list"
        ? 16
        : intents.length >= 2
          ? 14
          : MAX_FACTS;
  const links: SearchBriefLink[] = [];
  const gaps: string[] = [];

  for (const s of sources) {
    if (s.ok && s.text.trim()) {
      const fmt = providerFormatters[s.provider] ?? formatGeneric;
      const cap = PROVIDER_FACT_CAPS[s.provider] ?? 4;
      let added = 0;
      for (const f of fmt(s.text)) {
        if (facts.length >= maxFacts || added >= cap) break;
        facts.push(`[${s.label}] ${f}`);
        added++;
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

  if (isIsraelCinemaNowQuery(query)) {
    const webSources = sources.filter(
      (s) =>
        (s.provider === "tavily" || s.provider === "scavio" || s.provider === "searxng") &&
        s.ok &&
        (s.webHits?.length || s.text.trim()),
    );
    const movies = extractCinemaMoviesFromSources(webSources, 6);
    if (movies.length) {
      facts.unshift(
        `[Cinema IL] ANSWER (now showing): ${movies
          .slice(0, 6)
          .map((m, i) => `${i + 1}. ${m.title}`)
          .join(" · ")}`,
      );
      for (const m of movies.slice(0, 4)) {
        facts.unshift(`[${m.source}] סרט: ${m.title}`);
      }
    }
    for (const src of webSources) {
      let addedListing = false;
      for (const hit of src.webHits ?? []) {
        const listing = hit.snippet?.trim();
        if (!listing || parseCinemaMoviesFromText(listing).length < 2) continue;
        try {
          const host = new URL(hit.url).hostname.replace(/^www\./, "");
          facts.unshift(
            `[${src.label}] רשימת קופה (${host}): ${listing.slice(0, CINEMA_LISTING_FACT_MAX)}`,
          );
        } catch {
          facts.unshift(
            `[${src.label}] רשימת קופה: ${listing.slice(0, CINEMA_LISTING_FACT_MAX)}`,
          );
        }
        addedListing = true;
        break;
      }
      if (addedListing) continue;
    }
  }

  const rankedFacts = rerankBriefFacts(facts, intents, query, answerShape).slice(0, maxFacts);

  return { facts: rankedFacts, links, gaps, intents };
};

const answerShapeInstructions = (
  shape?: AnswerShape,
  intents?: SearchIntent[],
  query = "",
): string[] => {
  if (!shape) return [];
  if (shape === "bullet_list" && intents?.includes("news")) {
    return ["ANSWER SHAPE: bullet_list — 5–8 Hebrew headline bullets from ALL RSS outlets in FACTS; translate English."];
  }
  if (shape === "bullet_list" && needsOpenWebEnrichment(query)) {
    if (isIsraelCinemaNowQuery(query) && /(?:תקציר|summary|עליל)/i.test(query)) {
      return [
        "ANSWER SHAPE: bullet_list — 3 Hebrew bullets; one per movie from ANSWER (now showing) / Cinema IL facts ONLY.",
        "Each bullet: «שם הסרט — תקציר שורה אחת»; use general knowledge ONLY for well-known films listed in FACTS; if plot unknown say «מוקרן כרגע — אין תקציר בדף הקופה».",
        "NO placeholders. NO homepage text. NO duplicate site names.",
      ];
    }
    return [
      "ANSWER SHAPE: bullet_list — use WEB FACTS (Tavily/SearXNG/Scavio) titles/snippets; one Hebrew bullet per movie/team/player; NO placeholders like [תקציר].",
    ];
  }
  const map: Record<AnswerShape, string> = {
    short_fact: "ANSWER SHAPE: short_fact — one crisp Hebrew sentence; lead with ANSWER line if present.",
    count: "ANSWER SHAPE: count — lead with a number (מטוסים/ספינות/AQI/מagnitude).",
    bullet_list: "ANSWER SHAPE: bullet_list — 2–4 short bullets in Hebrew.",
    overview: "ANSWER SHAPE: overview — 3–4 sentences summarizing key sources.",
  };
  return [map[shape]];
};

export const formatSearchBriefContext = (
  brief: SearchBrief,
  query: string,
  maxChars = 900,
  sources: SearchSourceResult[] = [],
  answerShape?: AnswerShape,
  regionLabel?: string,
): string => {
  const lines = [
    `[SEARCH BRIEF — live data for: ${truncate(query, 80)}]`,
    "Use ONLY facts below. Cite source labels. Do NOT invent numbers, names, or URLs.",
    "If ANSWER line exists — lead with it in Hebrew. If GAPS exist — mention them honestly first.",
    "If DATA AGE exists — say «עדכון אחרון מ-…», NOT «כרגע» as intraday.",
    brief.intents.includes("news")
      ? "For NEWS: write 5–8 short Hebrew bullets — one headline per outlet from ANSWER (news) block; translate English. No single-line answers."
      : "Max 4 sentences. No philosophy. No follow-up questions.",
  ];
  if (regionLabel) {
    lines.splice(2, 0, `SHARED REGION: ${regionLabel} — compare sources for this area.`);
  }
  lines.splice(2, 0, ...answerShapeInstructions(answerShape, brief.intents, query));
  if (shouldBuildCrossSourceCorrelation(query, brief.intents)) {
    const metrics = extractCrossSourceMetrics(sources, regionLabel);
    const correlation = buildCrossSourceCorrelationLines(query, metrics, brief.intents);
    if (correlation.length) {
      lines.splice(2, 0, ...correlation);
    }
  }
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
  if (isIsraelCinemaNowQuery(query) && brief.facts.some((f) => /ANSWER \(now showing\)/i.test(f))) {
    const answerFact =
      brief.facts.find((f) => /^\[Cinema IL\]\s*ANSWER \(now showing\)/i.test(f)) ??
      brief.facts.find((f) => /ANSWER \(now showing\)/i.test(f));
    const listingFacts = brief.facts.filter((f) => /רשימת קופה/i.test(f)).slice(0, 2);
    if (answerFact) {
      lines.splice(2, 0, answerFact.replace(/^\[[^\]]+\]\s*/, ""));
    }
    for (const listing of listingFacts) {
      lines.splice(3, 0, listing.replace(/^\[[^\]]+\]\s*/, ""));
    }
  }
  if (brief.intents.includes("ships") && brief.facts.some((f) => /ANSWER \(ships live\)|ספינות בטווח:/.test(f))) {
    const countFact =
      brief.facts.find((f) => /ANSWER \(ships live\)/.test(f)) ??
      brief.facts.find((f) => /ספינות בטווח:/.test(f));
    const gapFact = brief.facts.find((f) => /^[^\n]*הערה:/.test(f));
    if (countFact) {
      const count = countFact.match(/ANSWER \(ships live\):\s*(\d+)/)?.[1] ?? countFact.match(/:\s*(\d+)/)?.[1];
      lines.splice(2, 0, `ANSWER (ships): ${count ?? "0"} אוניות עם AIS חי`);
    }
    if (gapFact && /Digitraffic|אין כיסוי|הדגמה|אין דיווח AIS/i.test(gapFact)) {
      lines.splice(3, 0, "GAPS: אין דיווח AIS לאזור — הספירה 0.");
    }
  }
  if (brief.intents.includes("aviation") && brief.facts.some((f) => /מטוסים בטווח:|כל\s+המטוסים:|סה"כ\s+\d+\s+מטוסים/i.test(f))) {
    const countFact =
      brief.facts.find((f) => /מטוסים בטווח:/.test(f)) ??
      brief.facts.find((f) => /כל\s+המטוסים:/.test(f)) ??
      brief.facts.find((f) => /סה"כ\s+\d+\s+מטוסים/.test(f));
    if (countFact) {
      const plain = countFact.replace(/^\[[^\]]+\]\s*/, "");
      const countLine = /מטוסים בטווח:/.test(plain)
        ? plain
        : /כל\s+המטוסים:/.test(plain)
          ? plain.replace(/כל\s+המטוסים:/, "מטוסים בטווח:")
          : plain.replace(/סה"כ\s+(\d+)\s+מטוסים/i, "מטוסים בטווח: $1");
      lines.splice(4, 0, `ANSWER (aircraft count): ${countLine}`);
    }
  }
  if (brief.intents.includes("earthquake") && brief.facts.some((f) => /סה"כ|M\d+\.\d/i.test(f))) {
    const eqLead =
      brief.facts.find((f) => /הרעידה/i.test(f)) ??
      brief.facts.find((f) => /סה"כ/i.test(f)) ??
      brief.facts.find((f) => /^\- M/.test(f.replace(/^\[[^\]]+\]\s*/, "")));
    if (eqLead) {
      lines.splice(4, 0, `ANSWER (earthquake): ${eqLead.replace(/^\[[^\]]+\]\s*/, "")}`);
    }
  }
  if (brief.intents.includes("earthquake") && brief.intents.includes("news")) {
    lines.splice(
      3,
      0,
      "SENSOR+RSS: USGS FACTS = magnitudes/locations/times; NEWS = media — correlate same region if possible; note alarming headlines.",
    );
  }
  if (brief.intents.includes("airquality") && brief.facts.some((f) => /ANSWER \(air quality\)|US AQI/i.test(f))) {
    const aqFact =
      brief.facts.find((f) => /ANSWER \(air quality\)/i.test(f)) ??
      brief.facts.find((f) => /US AQI/i.test(f));
    if (aqFact) lines.splice(4, 0, `ANSWER (air quality): ${aqFact.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  if (brief.intents.includes("marine-infra") && brief.facts.some((f) => /תשתיות ימיות/.test(f))) {
    const infraFact = brief.facts.find((f) => /תשתיות ימיות/.test(f));
    if (infraFact) lines.splice(4, 0, `ANSWER (marine infra): ${infraFact.replace(/^\[[^\]]+\]\s*/, "")}`);
  }
  const newsHeadline = brief.facts.find((f) => /^\[חדשות/.test(f) && /\d+\./.test(f));
  const newsHeadlines = sources
    .filter((s) => s.provider === "grovee-news" && s.ok && s.text.trim())
    .map((s) => {
      const m =
        s.text.match(/ANSWER \(headline\):\s*\[([^\]]+)\]\s*(.+)/) ??
        s.text.match(/^\[([^\]]+)\]\s*1\.\s*(.+)/m);
      const label = s.label.replace(/^חדשות \(/, "").replace(/\)$/, "");
      return m ? `[${m[1] ?? label}] ${m[2].trim()}` : null;
    })
    .filter((h): h is string => !!h);
  if (brief.intents.includes("news") && newsHeadlines.length) {
    lines.splice(
      4,
      0,
      `ANSWER (news — summarize ALL outlets below in Hebrew; English headlines → translate):`,
      ...newsHeadlines.slice(0, 10).map((h) => `- ${h}`),
    );
  } else if (brief.intents.includes("news") && newsHeadline) {
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
    if (regionLabel) {
      lines.push(`CROSS-SOURCE GEO: All sources scoped to «${regionLabel}» when possible.`);
    }
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
