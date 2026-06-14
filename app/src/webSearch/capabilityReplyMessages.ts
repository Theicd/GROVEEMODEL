import {
  isAviationQuery,
  isIssQuery,
  isMarineInfraQuery,
  isNewsQuery,
  isGitHubPopularQuery,
  isShipsQuery,
  isStarlinkCountQuery,
  isStarlinkRegionalQuery,
  isUnsupportedLiveQuery,
  isWorldOverviewQuery,
} from "./intents";
import { formatDataAgeForSource } from "./dataAge";
import { fallbackFromLiveWorldSnapshot } from "../liveWorld/snapshotFallback";
import { issSearchResultFromLiveWorld } from "../liveWorld/issSnapshot";
import { getCachedLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import { buildMilitaryAviationText } from "../liveWorld/militaryAviation";
import { buildMarineLiveReply } from "./marineReplyMessages";
import type { SearchIntent, SearchSourceResult } from "./types";

const PROVIDER_INTROS: Partial<Record<SearchSourceResult["provider"], string>> = {
  "news-rss": "לפי עדכוני RSS:",
  "wikidata-gov": "לפי Wikidata (ממשל):",
  "frankfurter-fx": "לפי Frankfurter (ECB):",
  "yahoo-finance": "לפי Yahoo Finance:",
  coingecko: "לפי CoinGecko:",
  "usgs-earthquake": "לפי USGS (רעידות אדמה):",
  "open-meteo": "לפי Open-Meteo (מזג אוויר):",
  "open-meteo-marine": "לפי Open-Meteo Marine (גלים):",
  "adsb-aviation": "לפי ADS-B חי:",
  "iss-tracker": "לפי מעקב ISS:",
  "gdacs-disasters": "לפי GDACS (אסונות טבע):",
  github: "לפי GitHub:",
  "huggingface-models": "לפי Hugging Face (מודלים):",
  "huggingface-datasets": "לפי Hugging Face (datasets):",
  celestrak: "לפי ספירת לוויינים:",
  "starlink-catalog": "לפי CelesTrak (Starlink):",
  "israel-alerts": "לפי פיקוד העורף:",
  "nominatim-places": "לפי OpenStreetMap:",
  "osrm-distance": "לפי חישוב מרחק:",
  "hacker-news": "לפי Hacker News:",
};

const UNSUPPORTED_REPLIES: Array<{ re: RegExp; text: string }> = [
  {
    re: /(?:שדה\s+התעופה|airport).*(?:העמוס|busiest)/i,
    text:
      "אין כרגע מקור חי בדפדפן לדירוג «השדה העמוס ביותר עכשיו». אפשר לראות מטוסים מעל אזור ספציפי (למשל ישראל או לונדון) — נסה «כמה מטוסים מעל ישראל?».",
  },
  {
    re: /(?:נמל\s+העמוס|busiest\s+port)/i,
    text:
      "דירוג «הנמל העמוס ביותר כרגע» אינו זמין במקור חי. אפשר לספור אוניות באזור נמל מוגדר — למשל «כמה אוניות במפרץ סואץ?».",
  },
  {
    re: /starlink.*(?:מעל|above|over|באזור|אירופ)/i,
    text:
      "מעקב Starlink לפי אזור אינו נתמך במקור חי בדפדפן כרגע. אפשר לספור את כל מאגר Starlink (CelesTrak) או לראות ISS ולוויינים על שכבת «לוויינים» בעולם חי.",
  },
  {
    re: /(?:אילו|which).*(?:starlink|לוויין)/i,
    text:
      "רשימת Starlink לפי אזור אינה זמינה בדפדפן. נסה «כמה לווייני Starlink פעילים כרגע?» לספירה גלובלית מ-CelesTrak.",
  },
  {
    re: /(?:קווי\s+רכבת|train\s+lines?\s+to|אילו\s+קווי\s+רכבת)/i,
    text:
      "לוחות זמנים וקווי רכבת (GTFS) אינם מחוברים. ניתן למצוא תחנת רכבת קרובה לשדה תעופה ולחשב מרחק/זמן משוער.",
  },
];

const formatGenericSource = (source: SearchSourceResult): string => {
  const intro = PROVIDER_INTROS[source.provider] ?? `לפי ${source.label}:`;
  const body = source.text.trim();
  if (!body) return "";

  const lines = body.split("\n").filter(Boolean);
  const dataAge = formatDataAgeForSource(source);

  if (source.provider === "wikidata-gov") {
    const pmAnswer = source.text.match(/ANSWER: ראש הממשלה \(Wikidata\): (.+)/);
    const pmRow = source.text.split("\n").find((l) => /^-\s/.test(l) && /ראש ממשלה|prime minister/i.test(l));
    const country =
      source.text.match(/^מדינה:\s*(.+)$/m)?.[1]?.replace(/\s*\(Wikidata[^)]*\)/i, "").trim() ?? "";
    const lead = pmAnswer
      ? `ראש ממשלת ${country || "בריטניה"} הוא ${pmAnswer[1].trim()}.`
      : pmRow
        ? `ראש ממשלת ${country || "המדינה"}: ${pmRow.replace(/^-\s*/, "").split("·")[0]?.trim()}.`
        : intro;
    return [lead, `Sources: ${source.label}`, `מקור: ${source.label}.`].join("\n");
  }

  if (source.provider === "frankfurter-fx") {
    const date = lines.find((l) => l.startsWith("תאריך:"))?.replace("תאריך:", "").trim();
    const rate = lines.find((l) => /1 USD =/.test(l));
    const lead = rate
      ? `שער הדולר מול השקל — עדכון אחרון מ-${date ?? "?"}: ${rate.replace(/^•\s*/, "")}`
      : intro;
    return [
      lead,
      ...(dataAge ? [dataAge] : []),
      `Sources: ${source.label}`,
      `מקור: ${source.label}.`,
    ].join("\n");
  }

  if (source.provider === "yahoo-finance") {
    const value = lines.find((l) => /:\s*[\d,.]+ points/i.test(l) || /S&P|מדד/i.test(l));
    const lead = value
      ? `ערך מדד S&P 500 — ${value.replace(/^•\s*/, "")}`
      : intro;
    return [
      lead,
      ...(dataAge ? [dataAge] : []),
      `Sources: ${source.label}`,
      `מקור: ${source.label}.`,
    ].join("\n");
  }

  if (source.provider === "news-rss") {
    const answer = lines.find((l) => l.startsWith("ANSWER (headline):"));
    const tagged = lines.filter((l) => /^\[(BBC|CNN|Reuters|Guardian|ynet)\]/i.test(l.trim()));
    const numbered = lines.filter((l) => /^\d+\.\s/.test(l.trim()));
    const headlineLines = tagged.length ? tagged : numbered;
    const lead = answer
      ? answer.replace("ANSWER (headline):", "הכותרת הראשית:")
      : headlineLines[0]
        ? `הכותרת הראשית: ${headlineLines[0].replace(/^\d+\.\s*/, "")}`
        : intro;
    const sourcesLine = lines.find((l) => l.startsWith("מקורות RSS"));
    const extras = headlineLines.slice(1, 6);
    return [
      lead,
      sourcesLine ?? "",
      extras.length ? "כותרות נוספות:" : "",
      ...extras,
      `Sources: ${source.label}`,
      "מקור: RSS — BBC · CNN · Reuters · Guardian (כותרות עולם).",
    ]
      .filter(Boolean)
      .join("\n");
  }

  if (source.provider === "open-meteo") {
    const temp = lines.find((l) => /טמפרatura|temperatur/i.test(l));
    const place = lines.find((l) => l.startsWith("מיקום:"));
    const condition = lines.find((l) => l.startsWith("מצב:"));
    const tempVal = temp?.match(/([\d.]+)\s*°C/i)?.[1];
    const lead =
      tempVal && place
        ? `הטמפרטורה ב${place.replace("מיקום:", "").trim()}: ${tempVal}°C`
        : temp ?? intro;
    const extras = [condition, temp !== lead ? temp : null].filter(Boolean) as string[];
    return [
      lead,
      ...extras.map((l) => `• ${l}`),
      `Sources: ${source.label}`,
      `מקור: ${source.label}.`,
    ].join("\n");
  }

  if (source.provider === "open-meteo-marine") {
    const wave = lines.find((l) => l.startsWith("גובה גל:"));
    const place = lines.find((l) => l.startsWith("מיקום:"));
    const waveVal = wave?.match(/([\d.]+)\s*m/)?.[1];
    const lead =
      waveVal && place
        ? `גובה הגלים מול ${place.replace("מיקום:", "").trim()}: ${waveVal} מטר`
        : wave ?? intro;
    return [lead, `Sources: ${source.label}`, `מקור: ${source.label}.`].join("\n");
  }

  if (source.provider === "iss-tracker") {
    const answer = lines.find((l) => l.startsWith("ANSWER (ISS position):"));
    const lat = lines.find((l) => l.startsWith("קו רוחב:"));
    const lon = lines.find((l) => l.startsWith("קו אורך:"));
    const alt = lines.find((l) => l.startsWith("גובה:"));
    const lead =
      answer?.replace("ANSWER (ISS position):", "מיקום ISS:") ??
      (lat && lon ? `תחנת החלל הבינלאומית: ${lat}, ${lon}${alt ? `, ${alt}` : ""}` : intro);
    return [lead, `Sources: ${source.label}`, `מקור: ${source.label}.`].join("\n");
  }

  if (source.provider === "starlink-catalog") {
    const answer = lines.find((l) => l.startsWith("ANSWER (Starlink active):"));
    const count = answer?.match(/:\s*(\d+)/)?.[1];
    const updated = lines.find((l) => l.startsWith("עודכן:"));
    const lead = count
      ? `לווייני Starlink פעילים בקטalog CelesTrak: ${count}`
      : intro;
    return [
      lead,
      updated ?? "",
      "Sources: Starlink (CelesTrak / עולם חי)",
      "מקור: CelesTrak GROUP=starlink — אותו מאגר TLE שמוגדר בעולם חי.",
    ]
      .filter(Boolean)
      .join("\n");
  }

  if (source.provider === "github") {
    const answer = lines.find((l) => l.startsWith("ANSWER (GitHub top):"));
    const filter = lines.find((l) => l.startsWith("סינון:"));
    const topLine = lines.find((l) => l.includes("הפרויקט הפופולרי") || /^1\.\s/.test(l));
    const lead = answer
      ? answer.replace("ANSWER (GitHub top):", "הפרויקט המוביל ב-GitHub:")
      : topLine ?? intro;
    const extras = lines
      .filter((l) => /^[2-4]\.\s/.test(l))
      .slice(0, 3);
    return [
      lead,
      filter ?? "",
      ...extras,
      "Sources: GitHub Repositories",
      "מקור: GitHub Search API — מאגרים פעילים עם push אחרון, ממוינים לפי כוכבים.",
    ]
      .filter(Boolean)
      .join("\n");
  }

  const bullets = lines.map((l) => (l.startsWith("- ") || /^\d+\./.test(l) ? l : `• ${l}`));
  const lead = bullets[0] ?? intro;

  return [
    lead,
    ...bullets.slice(1),
    ...(dataAge ? [dataAge] : []),
    `Sources: ${source.label}`,
    `מקור: ${source.label}.`,
  ].join("\n");
};

const parseAircraftCountLine = (text: string): { count: string; region: string } | null => {
  const countM = text.match(/מטוסים (?:בטווח|באוויר)[^:\n]*:\s*(\d+)/i);
  if (!countM) return null;
  const regionM = text.match(/^אזור:\s*(.+)$/m);
  return { count: countM[1], region: regionM?.[1]?.trim() ?? "האזור המבוקש" };
};

const buildIssLiveReply = (query: string, sources: SearchSourceResult[]): string | null => {
  if (!isIssQuery(query)) return null;

  const live = sources.find((s) => s.provider === "iss-tracker" && s.ok && s.text.trim());
  if (live) {
    const lines = live.text.trim().split("\n").filter(Boolean);
    const answer = lines.find((l) => l.startsWith("ANSWER (ISS position):"));
    const lat = lines.find((l) => l.startsWith("קו רוחב:"));
    const lon = lines.find((l) => l.startsWith("קו אורך:"));
    const alt = lines.find((l) => l.startsWith("גובה:"));
    return [
      "לפי מעקב ISS (עולם חי / wheretheiss):",
      answer?.replace("ANSWER (ISS position):", "מיקום ISS:") ??
        [lat, lon, alt].filter(Boolean).join(" · "),
      lat,
      lon,
      alt,
      lines.find((l) => l.startsWith("עודכן:")) ?? "",
      "Sources: תחנת חלל (ISS / עולם חי)",
      "מקור: עולם חי / wheretheiss.",
    ]
      .filter(Boolean)
      .join("\n");
  }

  const cached = issSearchResultFromLiveWorld(query);
  if (cached) return formatGenericSource(cached);

  const fb = fallbackFromLiveWorldSnapshot(query, ["satellite"]);
  if (fb) return formatGenericSource(fb);

  return null;
};

const buildGitHubLiveReply = (query: string, sources: SearchSourceResult[]): string | null => {
  if (!isGitHubPopularQuery(query)) return null;
  const live = sources.find((s) => s.provider === "github" && s.ok && s.text.trim());
  if (!live) return null;
  return formatGenericSource(live);
};

const buildStarlinkLiveReply = (query: string, sources: SearchSourceResult[]): string | null => {
  if (!isStarlinkCountQuery(query) || isStarlinkRegionalQuery(query)) return null;

  const live = sources.find((s) => s.provider === "starlink-catalog" && s.ok && s.text.trim());
  if (live) return formatGenericSource(live);

  return null;
};

const buildMilitaryAviationLiveReply = (query: string, sources: SearchSourceResult[]): string | null => {
  if (!/\bawacs\b|צבאי|military|תדלוק|tanker|מודיעין/i.test(query)) return null;

  const snap = getCachedLiveWorldSnapshot(120_000);
  if (snap?.aviation?.items?.length) {
    const text = buildMilitaryAviationText(query, snap);
    if (text) {
      const answer = text.match(/ANSWER \(AWACS\):[^\n]+/)?.[0];
      const awacsCount = text.match(/AWACS\?: (\d+)/)?.[1] ?? text.match(/מועמדים ל-AWACS[^:]*: (\d+)/)?.[1];
      return [
        "לפי עולם חי / ADS-B:",
        answer ?? (awacsCount != null ? `מועמדים ל-AWACS: ${awacsCount}` : text.split("\n").slice(0, 4).join("\n")),
        "Sources: תעופה (עולם חי / ADS-B)",
        "מקור: עולם חי — זיהוי צבאי/AWACS heuristic כמו על הגלובוס.",
      ].join("\n");
    }
  }

  const src = sources.find((s) => s.provider === "adsb-aviation" && s.ok && s.text.trim());
  if (!src) return null;
  const answer = src.text.match(/ANSWER \(AWACS\):[^\n]+/)?.[0];
  const milLine = src.text.match(/מועמדים ל-AWACS[^:]*: (\d+)/)?.[1]
    ?? src.text.match(/מטוסים צבאיים \(heuristic\): (\d+)/)?.[1];
  if (/\bawacs\b/i.test(query)) {
    return [
      "לפי ADS-B חי (זיהוי AWACS heuristic):",
      answer ?? (milLine != null ? `מועמדים ל-AWACS: ${milLine}` : src.text.split("\n").slice(0, 3).join("\n")),
      "Sources: תעופה (ADS-B)",
      "מקור: ADS-B — heuristic כמו שכבת תעופה בעולם חי.",
    ].join("\n");
  }
  if (/צבאי|military/i.test(query)) {
    return [
      "לפי ADS-B חי:",
      src.text.split("\n").slice(0, 5).join("\n"),
      "Sources: תעופה (ADS-B)",
      "מקור: ADS-B.",
    ].join("\n");
  }
  return null;
};

const buildAviationLiveReply = (
  query: string,
  sources: SearchSourceResult[],
): string | null => {
  if (!isAviationQuery(query) && !/כמה\s+מטוס/i.test(query)) return null;
  const src = sources.find((s) => s.provider === "adsb-aviation" && s.ok && s.text.trim());
  if (!src) return null;
  const parsed = parseAircraftCountLine(src.text);
  if (!parsed) return null;
  const intro = /כמה|how\s+many/i.test(query)
    ? `לפי ADS-B חי לגבי ${parsed.region}:`
    : `לפי ADS-B חי סביב ${parsed.region}:`;
  const samples = src.text
    .split("\n")
    .filter((l) => /^\d+\.\s/.test(l.trim()))
    .slice(0, 3)
    .map((l) => `• ${l.trim().replace(/^\d+\.\s*/, "")}`);
  return [
    intro,
    `מטוסים בטווח: ${parsed.count}`,
    ...(samples.length ? ["דוגמאות:", ...samples] : []),
    "Sources: תעופה (ADS-B)",
    "מקור: תעופה (ADS-B).",
  ].join("\n");
};

const buildOverviewReply = (sources: SearchSourceResult[]): string | null => {
  const ok = sources.filter((s) => s.ok && s.text.trim());
  if (ok.length < 2) return null;
  const sections = ok.slice(0, 6).map((s) => {
    const firstLines = s.text.trim().split("\n").slice(0, 4).join("\n");
    return `**${s.label}**\n${firstLines}`;
  });
  return [
    "סקירת מצב עולם (נתונים חיים ממקורות מרובים):",
    "",
    ...sections,
    "",
    "הערה: לשילוב ויזואלי — «הצג על הגלובוס» או פתח את פאנל עולם חי.",
  ].join("\n");
};

const buildUnsupportedReply = (query: string): string | null => {
  for (const { re, text } of UNSUPPORTED_REPLIES) {
    if (re.test(query)) return text;
  }
  if (isUnsupportedLiveQuery(query)) {
    return "סוג הנתון המבוקש אינו נתמך במקור חי בדפדפן כרגע. נסה ניסוח ספציפי יותר (אזור, נמל, מדינה).";
  }
  return null;
};

const pickPrimarySource = (
  query: string,
  intents: SearchIntent[],
  sources: SearchSourceResult[],
): SearchSourceResult | null => {
  const ok = sources.filter((s) => s.ok && s.text.trim());
  if (!ok.length) return null;

  if (isShipsQuery(query) && !isMarineInfraQuery(query)) {
    return ok.find((s) => s.provider === "ais-ships") ?? ok[0];
  }
  if (isMarineInfraQuery(query)) {
    return ok.find((s) => s.provider === "osm-overpass-marine") ?? ok[0];
  }
  if (isNewsQuery(query)) return ok.find((s) => s.provider === "news-rss") ?? ok[0];
  if (intents.includes("government")) return ok.find((s) => s.provider === "wikidata-gov") ?? ok[0];
  if (intents.includes("currency")) return ok.find((s) => s.provider === "frankfurter-fx") ?? ok[0];
  if (intents.includes("market")) return ok.find((s) => s.provider === "yahoo-finance") ?? ok[0];
  if (intents.includes("crypto")) return ok.find((s) => s.provider === "coingecko") ?? ok[0];
  if (isAviationQuery(query) || intents.includes("aviation")) {
    return ok.find((s) => s.provider === "adsb-aviation") ?? ok[0];
  }
  if (intents.includes("earthquake")) return ok.find((s) => s.provider === "usgs-earthquake") ?? ok[0];
  if (intents.includes("weather")) return ok.find((s) => s.provider === "open-meteo") ?? ok[0];
  if (intents.includes("marine")) return ok.find((s) => s.provider === "open-meteo-marine") ?? ok[0];
  if (intents.includes("satellite") && isStarlinkCountQuery(query)) {
    return ok.find((s) => s.provider === "starlink-catalog") ?? ok[0];
  }
  if (intents.includes("satellite") && /\biss\b|תחנת\s+(?:ה)?חלל|החלל\s+הבינלאומ/i.test(query)) {
    return ok.find((s) => s.provider === "iss-tracker") ?? ok.find((s) => s.provider === "celestrak") ?? ok[0];
  }
  if (intents.includes("satellite")) return ok.find((s) => s.provider === "iss-tracker") ?? ok[0];
  if (intents.includes("disaster")) return ok.find((s) => s.provider === "gdacs-disasters") ?? ok[0];
  if (intents.includes("github")) return ok.find((s) => s.provider === "github") ?? ok[0];
  if (intents.includes("huggingface")) {
    return ok.find((s) => s.provider === "huggingface-models") ?? ok[0];
  }
  if (intents.includes("places")) return ok.find((s) => s.provider === "nominatim-places") ?? ok[0];
  if (intents.includes("distance")) return ok.find((s) => s.provider === "osrm-distance") ?? ok[0];
  if (intents.includes("hackernews")) return ok.find((s) => s.provider === "hacker-news") ?? ok[0];
  return ok[0];
};

/** Fixed Hebrew when live providers returned data — avoids LLM ignoring SEARCH BRIEF. */
export function buildCapabilityLiveReply(
  query: string,
  intents: SearchIntent[],
  sources: SearchSourceResult[],
): string | null {
  const q = query.trim();
  if (!q) return null;

  const unsupportedEarly = buildUnsupportedReply(q);
  const unsupported = unsupportedEarly;
  if (unsupported && !sources.some((s) => s.ok && s.text.trim())) {
    return unsupported;
  }

  const marine = buildMarineLiveReply(q, intents, sources);
  if (marine) return marine;

  const militaryAv = buildMilitaryAviationLiveReply(q, sources);
  if (militaryAv) return militaryAv;

  const issReply = buildIssLiveReply(q, sources);
  if (issReply) return issReply;

  const githubReply = buildGitHubLiveReply(q, sources);
  if (githubReply) return githubReply;

  const starlinkReply = buildStarlinkLiveReply(q, sources);
  if (starlinkReply) return starlinkReply;

  const aviation = buildAviationLiveReply(q, sources);
  if (aviation) return aviation;

  if (isWorldOverviewQuery(q)) {
    const overview = buildOverviewReply(sources);
    if (overview) return overview;
  }

  const primary = pickPrimarySource(q, intents, sources);
  if (primary) return formatGenericSource(primary);

  if (isIssQuery(q)) {
    return "לא הצלחתי לטעון מיקום ISS — פתח REALITY LIVE (עולם חי) כמה שניות ונסה שוב, או «הצג על הגלובוס».";
  }

  if (unsupported) return unsupported;
  return null;
}

export { buildMarineLiveReply, formatGenericSource, pickPrimarySource };
