import {
  isAviationQuery,
  isAirQualityQuery,
  isEarthquakeQuery,
  isGitHubPopularQuery,
  isIssQuery,
  isMarineInfraQuery,
  isMarineQuery,
  isNewsQuery,
  isPriceQuery,
  isProductsQuery,
  isShipsQuery,
  isStarlinkCountQuery,
  isStarlinkRegionalQuery,
  isUnsupportedLiveQuery,
  isWeatherQuery,
  isWorldOverviewQuery,
} from "./intents";
import { formatDataAgeForSource } from "./dataAge";
import { formatProductPriceSummary, isCheapersalConfigured } from "./providers/cheapersalPrices";
import { fallbackFromLiveWorldSnapshot } from "../liveWorld/snapshotFallback";
import { issSearchResultFromLiveWorld } from "../liveWorld/issSnapshot";
import { getCachedLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import { buildMilitaryAviationText } from "../liveWorld/militaryAviation";
import { buildMarineLiveReply } from "./marineReplyMessages";
import { detectImpossiblePlace } from "./entityValidation";
import { isCrossSourceQuery } from "./crossSourceIntents";
import { isTopicalOverviewRouting } from "./topicalEnrichment";
import { isGeneralNewsDigestQuery } from "./queryExtract";
import {
  buildCrossSourceCorrelationLines,
  extractCrossSourceMetrics,
} from "./crossSourceCorrelation";
import { agentDebugLog } from "../debugAgentLog";
import { buildNewsPanelGuideReply } from "../groveeNews/newsPanelGuideReply";
import type { AnswerShape, SearchIntent, SearchSourceResult } from "./types";

const PROVIDER_INTROS: Partial<Record<SearchSourceResult["provider"], string>> = {
  "grovee-news": "לפי GROVEE NEWS (מאגר מקומי):",
  "wikidata-gov": "לפי Wikidata (ממשל):",
  "frankfurter-fx": "לפי Frankfurter (ECB):",
  "yahoo-finance": "לפי Yahoo Finance:",
  coingecko: "לפי CoinGecko:",
  searxng: "לפי SearXNG:",
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
  "israeli-products": "לפי Cheapersal (מחירי סופרמרקטים בישראל):",
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

  if (source.provider === "israeli-products") {
    const hits = source.productHits ?? [];
    const priced = hits.filter((h) => h.priceNis != null);
    if (priced.length) {
      const top = priced[0];
      const lead = `${top.title}: ${formatProductPriceSummary(top)}.`;
      const extras = priced.slice(1, 4).map((h) => `• ${h.title} — ${formatProductPriceSummary(h)}`);
      return [
        intro,
        lead,
        extras.length ? "אפשרויות נוספות:" : "",
        ...extras,
        `Sources: ${source.label} · Cheapersal`,
        `מקור: השוואת מחירים מ-30+ רשתות סופר בישראל.`,
      ]
        .filter(Boolean)
        .join("\n");
    }
    if (hits.length) {
      const rows = hits.slice(0, 4).map((h, i) => `${i + 1}. ${h.title} [${h.barcode}]`);
      const answer = lines.find((l) => l.startsWith("ANSWER:"));
      const note = lines.find((l) => l.startsWith("הערה:"));
      return [
        intro,
        answer?.replace(/^ANSWER:\s*/, "") ?? `נמצאו ${hits.length} מוצרים:`,
        ...rows,
        note ?? "",
        `Sources: ${source.label}`,
      ]
        .filter(Boolean)
        .join("\n");
    }
    const answer = lines.find((l) => l.startsWith("ANSWER:"));
    if (answer) {
      return [intro, answer.replace(/^ANSWER:\s*/, ""), `Sources: ${source.label}`].join("\n");
    }
  }

  if (source.provider === "grovee-news") {
    const answer = lines.find((l) => l.startsWith("ANSWER (headline):"));
    const tagged = lines.filter((l) => /^\[[^\]]+\]\s*\d+\./i.test(l.trim()));
    const numbered = lines.filter((l) => /^\d+\.\s/.test(l.trim()));
    const headlineLines = tagged.length ? tagged : numbered;
    const lead = answer
      ? answer.replace("ANSWER (headline):", "הכותרת הראשית:")
      : headlineLines[0]
        ? `הכותרת הראשית: ${headlineLines[0].replace(/^\d+\.\s*/, "")}`
        : intro;
    const extras = headlineLines.slice(1, 6);
    return [
      lead,
      extras.length ? "כותרות נוספות:" : "",
      ...extras,
      `Sources: ${source.label}`,
      "מקור: GROVEE NEWS.",
    ]
      .filter(Boolean)
      .join("\n");
  }

  if (source.provider === "open-meteo") {
    const temp = lines.find((l) => /טמפר(?:atura|טור)/i.test(l));
    const place = lines.find((l) => l.startsWith("מיקום:"));
    const condition = lines.find((l) => l.startsWith("מצב:"));
    const humidity = lines.find((l) => l.startsWith("לחות:"));
    const wind = lines.find((l) => l.startsWith("רוח:"));
    const tempVal = temp?.match(/([\d.-]+)\s*°C/i)?.[1];
    const placeName = place?.replace("מיקום:", "").trim() ?? "";
    const conditionText = condition?.replace("מצב:", "").trim();
    const lead =
      tempVal && placeName
        ? `כרגע ב${placeName}: ${tempVal}°C${conditionText ? `, ${conditionText}` : ""}`
        : temp ?? intro;
    const extras = [humidity, wind].filter(Boolean) as string[];
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

  if (source.provider === "nominatim-places") {
    const first = lines.find((l) => /^\d+\./.test(l.trim()));
    const note = lines.find((l) => l.startsWith("הערה:"));
    const lead = first
      ? `תחנת הרכבת הקרובה לשדה התעופה: ${first.replace(/^\d+\.\s*/, "").split("\n")[0]?.trim()}`
      : intro;
    return [
      lead,
      note ?? "",
      `Sources: ${source.label}`,
      `מקור: ${source.label}.`,
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
  const countM =
    text.match(/מטוסים (?:בטווח|באוויר)[^:\n]*:\s*(\d+)/i) ??
    text.match(/סה[״"']?כ\s+(\d+)\s+מטוסים/i) ??
    text.match(/כל\s+המטוסים:\s*(\d+)/i);
  if (!countM) return null;
  const regionM =
    text.match(/^אזור:\s*(.+)$/m) ??
    text.match(/מקור:\s*עולם חי \/ ADS-B \((.+?)\)/i);
  return { count: countM[1], region: regionM?.[1]?.trim() ?? "האזור המבוקש" };
};

const buildEarthquakeLiveReply = (query: string, sources: SearchSourceResult[]): string | null => {
  if (!isEarthquakeQuery(query)) return null;
  const eq = sources.find((s) => s.provider === "usgs-earthquake" && s.ok && s.text.trim());
  if (!eq) return null;

  const lines = eq.text.trim().split("\n").filter(Boolean);
  const quakes = lines.filter((l) => l.trim().startsWith("- M"));
  const header = lines.find((l) => /^סה"כ/i.test(l.trim()));
  const lead =
    lines.find((l) => /^הרעידה/i.test(l.trim())) ??
    lines.find((l) => /^הרעידה החזקה/i.test(l.trim()));

  if (!quakes.length && /אין רעידות|לא נמצאו רעידות/i.test(eq.text)) {
    return [
      lines[0] ?? "אין רעידות אדמה מדווחות בתקופה האחרונה (USGS).",
      `Sources: ${eq.label}`,
      "מקור: USGS Earthquake Hazards Program.",
    ].join("\n");
  }

  return [
    "רעידות אדמה אחרונות (USGS):",
    lead ?? "",
    header ?? "",
    quakes.length ? "הגדולות:" : "",
    ...quakes.slice(0, 6).map((q) => `• ${q.replace(/^\-\s*/, "")}`),
    `Sources: ${eq.label}`,
    "מקור: USGS Earthquake Hazards Program.",
  ]
    .filter(Boolean)
    .join("\n");
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

const formatCrossSourceCanned = (
  correlationLines: string[],
  metrics: ReturnType<typeof extractCrossSourceMetrics>,
  sources: SearchSourceResult[],
  answerShape?: AnswerShape,
): string => {
  const labels = sources.map((s) => s.label).join(", ");
  const synthesis = correlationLines[0]?.replace(/^CORRELATION(?:: GEO)?:\s*/, "") ?? "";

  if (answerShape === "count") {
    const nums: string[] = [];
    if (metrics.aviation) nums.push(`${metrics.aviation.count} מטוסים`);
    if (metrics.ships) nums.push(`${metrics.ships.count} ספינות`);
    if (metrics.airQuality?.aqi != null) nums.push(`AQI ${metrics.airQuality.aqi}`);
    if (metrics.weather?.windKmh != null) nums.push(`רוח ${metrics.weather.windKmh} km/h`);
    return [nums.length ? nums.join(" · ") : synthesis, synthesis, `Sources: ${labels}`]
      .filter(Boolean)
      .join("\n");
  }

  if (answerShape === "bullet_list") {
    const bullets: string[] = [];
    if (metrics.weather) {
      bullets.push(
        `• מז"א: ${metrics.weather.condition ?? "—"}${metrics.weather.windKmh ? `, רוח ${metrics.weather.windKmh} km/h` : ""}`,
      );
    }
    if (metrics.aviation) bullets.push(`• ADS-B: ${metrics.aviation.count} מטוסים`);
    if (metrics.ships) bullets.push(`• AIS: ${metrics.ships.count} ספינות`);
    if (metrics.airQuality?.aqi != null) bullets.push(`• איכות אוויר: US AQI ${metrics.airQuality.aqi}`);
    return [synthesis, ...bullets, `Sources: ${labels}`].filter(Boolean).join("\n");
  }

  if (answerShape === "overview") {
    const sections = sources.slice(0, 4).map((s) => {
      const preview = s.text.trim().split("\n").slice(0, 3).join(" · ");
      return `${s.label}: ${preview}`;
    });
    return [synthesis, ...sections, `Sources: ${labels}`].join("\n");
  }

  return [synthesis, `Sources: ${labels}`].join("\n");
};

/** Canned cross-source reply when correlation + 2+ live sources (Phase 5). */
export const buildCrossSourceLiveReply = (
  query: string,
  intents: SearchIntent[],
  sources: SearchSourceResult[],
  answerShape?: AnswerShape,
  regionLabel?: string,
): string | null => {
  if (!isCrossSourceQuery(query) && intents.length < 2) return null;

  const ok = sources.filter((s) => s.ok && s.text.trim());
  if (ok.length < 2) return null;

  const metrics = extractCrossSourceMetrics(sources, regionLabel);
  let correlation = buildCrossSourceCorrelationLines(query, metrics, intents);

  if (!correlation.length && /^האם\s+/i.test(query)) {
    const eq = ok.find((s) => s.provider === "usgs-earthquake");
    if (eq && /אין רעידות|לא נמצאו רעידות/i.test(eq.text)) {
      correlation = ["CORRELATION: לא — אין רעידות אדמה מדווחות ב-24 שעות (USGS)."];
    } else if (metrics.aviation && metrics.weather) {
      correlation = [
        `CORRELATION GEO: מז"א ${metrics.weather.condition ?? "—"} · ADS-B ${metrics.aviation.count} מטוסים.`,
      ];
    } else if (ok.length >= 2) {
      const preview = ok
        .slice(0, 3)
        .map((s) => `${s.label}: ${s.text.trim().split("\n")[0] ?? ""}`)
        .join(" · ");
      correlation = [`CORRELATION: ${preview}`];
    }
  }

  if (!correlation.length) return null;

  if (
    answerShape === "overview" ||
    /^האם\s+/i.test(query) ||
    /קשר\s+בין|הצלב|compare|yes\s+or\s+no/i.test(query)
  ) {
    return formatCrossSourceCanned(correlation, metrics, ok, answerShape ?? "short_fact");
  }

  return null;
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

const pickBestNewsSource = (_query: string, sources: SearchSourceResult[]): SearchSourceResult | null => {
  const news = sources.filter((s) => s.provider === "grovee-news" && s.ok && s.text.trim());
  return news[0] ?? null;
};

function countNewsCardsInBrief(text: string): number {
  const rows = text.split("\n").filter((l) => /^\[[^\]]+\]\s*\d+\./.test(l.trim()));
  return rows.length || (text.includes("ANSWER (headline):") ? 1 : 0);
}

/** Headline reply from GROVEE NEWS engine. */
export const buildNewsAggregatedReply = (
  query: string,
  sources: SearchSourceResult[],
): string | null => {
  const news = sources.filter((s) => s.provider === "grovee-news" && s.ok && s.text.trim());
  if (!news.length) return null;

  const one = pickBestNewsSource(query, sources);
  return one ? formatGenericSource(one) : null;
};

const buildProductsPriceReply = (
  query: string,
  sources: SearchSourceResult[],
): string | null => {
  const prod = sources.find((s) => s.provider === "israeli-products");
  if (!prod) return null;

  if (!prod.ok && prod.error?.trim()) {
    if (!isPriceQuery(query) && !isProductsQuery(query)) return null;
    return [
      prod.error.trim(),
      isCheapersalConfigured()
        ? "ייתכן שהמוצר לא נמצא במאגר המחירים — נסה שם מדויק יותר (למשל «חלב תנובה 3%»)."
        : "קבל מפתח חינמי ב-cheapersal.co.il/developers והוסף CHEAPERSAL_API_KEY ל-app/.env.",
      "Sources: מוצרי סופר · ישראל",
    ].join("\n");
  }

  if (prod.ok && (prod.productHits?.length || prod.text.trim())) {
    return formatGenericSource(prod);
  }

  const priced = (prod.productHits ?? []).filter((h) => h.priceNis != null);
  if (!priced.length) {
    if (!isPriceQuery(query)) return null;
    return [
      "לא הצלחתי להביא מחירים מסופרמרקטים לשאלה הזו.",
      isCheapersalConfigured()
        ? "ייתכן שהמוצר לא נמצא במאגר — נסה שם מדויק יותר או ברקוד."
        : "הוסף CHEAPERSAL_API_KEY ב-app/.env (חינם: cheapersal.co.il/developers).",
      "Sources: מוצרי סופר · ישראל",
    ].join("\n");
  }

  return formatGenericSource({ ...prod, productHits: priced });
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
  if (isNewsQuery(query)) {
    const best = pickBestNewsSource(query, ok);
    if (best) return best;
  }
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
  if (intents.includes("products") || isPriceQuery(query)) {
    return ok.find((s) => s.provider === "israeli-products") ?? ok[0];
  }
  return ok[0];
};

export type CapabilityLiveReplyOptions = {
  answerShape?: AnswerShape;
  regionLabel?: string;
};

const STRUCTURED_LIVE_INTENTS = new Set<SearchIntent>([
  "weather",
  "airquality",
  "marine",
  "currency",
  "earthquake",
  "worldtime",
  "distance",
  "places",
  "holiday",
  "country",
  "government",
  "crypto",
  "market",
  "aviation",
  "ships",
  "marine-infra",
  "satellite",
  "disaster",
  "spaceweather",
  "github",
  "hackernews",
  "huggingface",
  "arxiv",
  "products",
]);

/** Bypass Gemma for structured live data — never for news (model must synthesize RSS). */
export function shouldDeliverStructuredLiveReply(
  query: string,
  intents: SearchIntent[],
  sources: SearchSourceResult[],
  cannedReply?: string | null,
): boolean {
  if (intents.includes("news") || isNewsQuery(query)) return false;
  if (cannedReply?.trim()) return true;
  if (!sources.some((s) => s.ok && s.text.trim())) return false;
  if (isWeatherQuery(query) || isAirQualityQuery(query) || isMarineQuery(query)) return true;
  if (isCrossSourceQuery(query) || isWorldOverviewQuery(query)) return true;
  if (intents.some((i) => STRUCTURED_LIVE_INTENTS.has(i))) return true;
  if (isAviationQuery(query) || isShipsQuery(query) || isIssQuery(query)) return true;
  if (isStarlinkCountQuery(query) || isGitHubPopularQuery(query)) return true;
  if (isPriceQuery(query)) return true;
  return false;
}

const stripScore = (line: string): string =>
  line.replace(/^\d+\.\s*/, "").replace(/\s*\(★[\d,]+\).*$/i, "").trim();

/** Bullet reply from ALL ok sources — no Gemma fluff for overview/news digest. */
export function buildOverviewMultiSourceReply(
  query: string,
  sources: SearchSourceResult[],
): string | null {
  const ok = sources.filter((s) => s.ok && s.text.trim());
  if (!ok.length) return null;

  const bullets: string[] = [];
  const labels: string[] = [];

  for (const ns of ok.filter((s) => s.provider === "grovee-news")) {
    const headline =
      ns.text.match(/ANSWER \(headline\):\s*\[([^\]]+)\]\s*(.+)/) ??
      ns.text.match(/^\[([^\]]+)\]\s*1\.\s*(.+)/m);
    if (headline) {
      bullets.push(`• [${headline[1]}] ${headline[2].trim()}`);
    } else {
      const row = ns.text.split("\n").find((l) => /^\[[^\]]+\]\s*\d+\./.test(l.trim()));
      if (row) bullets.push(`• ${row.trim()}`);
    }
    labels.push(ns.label.replace(/^חדשות \(/, "").replace(/\)$/, ""));
  }

  const hn = ok.find((s) => s.provider === "hacker-news");
  if (hn) {
    for (const line of hn.text.split("\n").filter((l) => /^\d+\./.test(l.trim())).slice(0, 3)) {
      bullets.push(`• [Hacker News] ${stripScore(line)}`);
    }
    labels.push("Hacker News");
  }

  const gh = ok.find((s) => s.provider === "github");
  if (gh) {
    for (const line of gh.text.split("\n").filter((l) => /^\d+\.|★/.test(l)).slice(0, 2)) {
      bullets.push(`• [GitHub] ${stripScore(line)}`);
    }
    labels.push("GitHub Repositories");
  }

  for (const s of ok.filter((x) => x.provider === "huggingface-models")) {
    const row = s.text.split("\n").find((l) => /^\d+\.|ANSWER/i.test(l));
    if (row) bullets.push(`• [Hugging Face] ${stripScore(row)}`);
    labels.push("Hugging Face");
  }

  for (const s of ok.filter((x) => x.provider === "arxiv")) {
    const row = s.text.split("\n").find((l) => /^\d+\./.test(l.trim()));
    if (row) bullets.push(`• [arXiv] ${stripScore(row)}`);
    labels.push("arXiv");
  }

  if (!bullets.length) return null;

  const intro = isGeneralNewsDigestQuery(query)
    ? "כותרות חדשות עדכניות ממקורות מרובים:"
    : isTopicalOverviewRouting(query)
      ? "עדכונים מהמקורות שנמצאו:"
      : "סיכום מהמקורות:";

  return [
    intro,
    ...bullets.slice(0, 8),
    `Sources: [${[...new Set(labels)].join(", ")}]`,
  ].join("\n");
}

/** Fixed Hebrew when live providers returned data — avoids LLM ignoring SEARCH BRIEF. */
export function buildCapabilityLiveReply(
  query: string,
  intents: SearchIntent[],
  sources: SearchSourceResult[],
  options?: CapabilityLiveReplyOptions,
): string | null {
  const q = query.trim();
  if (!q) return null;
  // #region agent log
  agentDebugLog("H4,H5", "capabilityReplyMessages.ts:buildCapabilityLiveReply", "canned reply evaluation started", {
    queryPreview: q.slice(0, 120),
    intents,
    answerShape: options?.answerShape,
    sourceLabels: sources.map((s) => ({ provider: s.provider, label: s.label, ok: s.ok, hasText: !!s.text.trim(), error: s.error?.slice(0, 120) })),
  });
  // #endregion

  const impossible = detectImpossiblePlace(q);
  if (impossible && /(?:מטוס|aircraft|weather|מזג|ספינ|ship)/i.test(q)) {
    return [
      `אין נתונים חיים ב-${impossible} — מקורות ADS-B, מזג אוויר ו-AIS זמינים רק לכדור הארץ.`,
      "נסה שאלה על אזור גיאוגרפי מוגדר (ישראל, ים תיכון, לונדון).",
      `Sources: (none — ${impossible})`,
    ].join("\n");
  }

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

  const earthquake = buildEarthquakeLiveReply(q, sources);
  if (earthquake) return earthquake;

  if (isWorldOverviewQuery(q)) {
    const overview = buildOverviewReply(sources);
    if (overview) return overview;
  }

  const crossSource = buildCrossSourceLiveReply(
    q,
    intents,
    sources,
    options?.answerShape,
    options?.regionLabel,
  );
  if (crossSource) return crossSource;

  const productsPrice = buildProductsPriceReply(q, sources);
  if (productsPrice) return productsPrice;

  if (isNewsQuery(q)) {
    const news = sources.find((s) => s.provider === "grovee-news");
    if (news) {
      const cardCount = news.ok ? countNewsCardsInBrief(news.text) : 0;
      return buildNewsPanelGuideReply(q, {
        mode: isTopicalOverviewRouting(q) ? "topics" : "search",
        cardCount,
      });
    }
    return null;
  }

  if (
    isTopicalOverviewRouting(q) ||
    options?.answerShape === "overview" ||
    (options?.answerShape === "bullet_list" && !isNewsQuery(q))
  ) {
    const multi = buildOverviewMultiSourceReply(q, sources);
    if (multi) return multi;
  }

  const primary = pickPrimarySource(q, intents, sources);
  if (primary) return formatGenericSource(primary);

  if (isIssQuery(q)) {
    return "לא הצלחתי לטעון מיקום ISS — פתח REALITY LIVE (עולם חי) כמה שניות ונסה שוב, או «הצג על הגלובוס».";
  }

  if (unsupported) return unsupported;
  return null;
}

/** Web fallback (SearXNG) failed or not configured — bypass LLM hallucination. */
export function buildWebFallbackNoDataReply(
  query: string,
  sources: SearchSourceResult[],
): string | null {
  const q = query.trim();
  if (!q) return null;
  if (sources.some((s) => s.ok && s.text.trim())) return null;

  const searx = sources.find((s) => s.provider === "searxng");
  if (!searx && sources.length > 0) return null;

  const err = searx?.error ?? "";
  const reason = err.includes("לא מוגדר")
    ? "חיפוש web (SearXNG) לא מוגדר — הוסף VITE_SEARXNG_URL בקובץ .env והפעל מחדש."
    : err.trim() || "חיפוש web נכשל (timeout / CORS / שרת לא זמין).";

  return [
    `לא הצלחתי להביא מידע עדכני מהרשת לשאלה הזו.`,
    reason,
    "Sources: (none — fetch failed)",
  ].join("\n");
}

export { buildMarineLiveReply, formatGenericSource, pickPrimarySource };
