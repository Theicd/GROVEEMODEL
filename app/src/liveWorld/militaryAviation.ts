import { classifyAircraft, isAwacsSuspect, isTankerSuspect } from "../realityData/aviationClassifier";
import type { LiveAviationItem, LiveWorldSnapshot } from "./types";

/** AWACS / airborne early-warning — English + common Hebrew spellings. */
export const isAwacsQuery = (query: string): boolean =>
  /\bawacs\b|אוואקס|א\.?ו\.?א\.?ק\.?ס|מערכת\s+(?:בקרה|מודיעין)\s+אווירית/i.test(query);

/** Refueling / tanker aircraft. */
export const isTankerAviationQuery = (query: string): boolean =>
  /תדלוק|tanker|refuel|מטוס(?:ים|י)?\s*(?:תדלוק|התדלוק)/i.test(query);

/** Military / AWACS / tanker aviation sub-queries (Hebrew + English). */
export const isMilitaryAviationQuery = (query: string): boolean =>
  isAwacsQuery(query) ||
  isTankerAviationQuery(query) ||
  /צבאי|צבאיים|military|מודיעין|מטוס(?:ים|י)?\s*(?:צבאיים|של\s+צבא)/i.test(query);

const sampleLinesFromText = (text: string, max = 3): string[] =>
  text
    .split("\n")
    .filter((l) => /^\d+\.\s/.test(l.trim()))
    .slice(0, max)
    .map((l) => `• ${l.trim().replace(/^\d+\.\s*/, "")}`);

/** User-facing canned reply for military aviation — count-first Hebrew. */
export const formatMilitaryAviationCannedReply = (
  query: string,
  sourceText: string,
): string | null => {
  if (!isMilitaryAviationQuery(query)) return null;

  const samples = sampleLinesFromText(sourceText);

  if (isAwacsQuery(query)) {
    const count =
      sourceText.match(/ANSWER \(AWACS\):\s*(\d+)/i)?.[1] ??
      sourceText.match(/מועמדים ל-AWACS[^:]*:\s*(\d+)/i)?.[1] ??
      sourceText.match(/(\d+)\s+AWACS\?/i)?.[1];
    if (count == null) return null;
    const n = parseInt(count, 10);
    if (n === 0) {
      return [
        "0 מטוסי AWACS מזוהים כרגע במעקב ADS-B (heuristic — לא כל AWACS משדר).",
        "Sources: תעופה (עולם חי / ADS-B)",
      ].join("\n");
    }
    const example =
      sourceText.match(/ANSWER \(AWACS\):\s*\d+\s+[^·\n]*·\s*([^·\n]+)/)?.[1]?.trim() ??
      samples[0]?.replace(/^•\s*/, "").split(" · ")[0]?.trim();
    return [
      `${n} מטוסי AWACS מסומנים כרגע במעקב ADS-B${example ? ` (למשל ${example})` : ""}.`,
      ...samples,
      "Sources: תעופה (עולם חי / ADS-B)",
    ]
      .filter(Boolean)
      .join("\n");
  }

  if (isTankerAviationQuery(query)) {
    const count =
      sourceText.match(/מועמדים לתדלוק[^:]*:\s*(\d+)/i)?.[1] ??
      sourceText.match(/(\d+)\s+תדלוק\?/i)?.[1];
    if (count == null) return null;
    const n = parseInt(count, 10);
    return [
      n === 0
        ? "0 מטוסי תדלוק מזוהים כרגע במעקב ADS-B (heuristic)."
        : `${n} מטוסי תדלוק במעקב ADS-B (heuristic).`,
      ...samples,
      "Sources: תעופה (עולם חי / ADS-B)",
    ]
      .filter(Boolean)
      .join("\n");
  }

  const count =
    sourceText.match(/מטוסים צבאיים \(heuristic\):\s*(\d+)/i)?.[1] ??
    sourceText.match(/מטוסים צבאיים[^:]*:\s*(\d+)/i)?.[1] ??
    sourceText.match(/(\d+)\s+צבאיים/i)?.[1];
  if (count == null) return null;
  const n = parseInt(count, 10);
  return [
    n === 0
      ? "0 מטוסים צבאיים מזוהים כרגע במעקב ADS-B (heuristic)."
      : `${n} מטוסים צבאיים במעקב ADS-B (heuristic).`,
    ...samples,
    "Sources: תעופה (עולם חי / ADS-B)",
  ]
    .filter(Boolean)
    .join("\n");
};

export const enrichAviationItem = (raw: {
  icao24?: string;
  callsign?: string;
  country?: string;
  category?: string | number;
  isMilitary?: boolean;
  milLabel?: string;
  geo?: { lat?: number; lon?: number; alt?: number };
  altitude?: number;
}): LiveAviationItem => {
  const cls = classifyAircraft(
    raw.icao24,
    raw.callsign,
    raw.country,
    raw.category,
    undefined,
  );
  const isMilitary = raw.isMilitary ?? cls.mil;
  const milLabel = raw.milLabel || cls.label;
  const awacsSuspect = cls.awacsSuspect || isAwacsSuspect(raw.callsign, milLabel, raw.category);
  const tankerSuspect = isTankerSuspect(raw.callsign, raw.category);
  return {
    icao24: raw.icao24,
    callsign: raw.callsign,
    country: raw.country,
    lat: raw.geo?.lat,
    lon: raw.geo?.lon,
    alt: raw.geo?.alt ?? raw.altitude,
    isMilitary,
    milLabel,
    awacsSuspect,
    tankerSuspect,
  };
};

export const summarizeAviationFromSnapshot = (
  snap: LiveWorldSnapshot,
): LiveWorldSnapshot["aviation"] | null => {
  const rawItems = snap.aviation?.items;
  if (!rawItems?.length) return snap.aviation ?? null;

  const items = rawItems.map((i) =>
    i.awacsSuspect != null ? i : enrichAviationItem(i),
  );
  const militaryCount = items.filter((i) => i.isMilitary).length;
  const awacsCount = items.filter((i) => i.awacsSuspect).length;
  const tankerCount = items.filter((i) => i.tankerSuspect).length;

  return {
    count: items.length,
    militaryCount,
    awacsCount,
    tankerCount,
    regionLabel: snap.aviation?.regionLabel ?? "עולם חי (ADS-B)",
    sample: items.slice(0, 5).map(formatAviationSampleLine),
    items,
  };
};

export const formatAviationSampleLine = (a: LiveAviationItem): string => {
  const tag = a.awacsSuspect ? " · AWACS?" : a.isMilitary ? " · צבאי" : "";
  const alt = a.alt != null ? `${Math.round(a.alt)}m` : "—";
  return `${a.callsign || a.icao24 || "—"} · ${alt}${tag}${a.milLabel ? ` · ${a.milLabel}` : ""}`;
};

export const buildMilitaryAviationText = (
  query: string,
  snap: LiveWorldSnapshot,
): string | null => {
  const av = summarizeAviationFromSnapshot(snap);
  if (!av?.items?.length) return null;

  const wantAwacs = isAwacsQuery(query);
  const wantTanker = isTankerAviationQuery(query);
  const wantMilitary = wantAwacs || wantTanker || /צבאי|צבאיים|military|מודיעין|מטוס(?:ים|י)?\s*(?:צבאיים|של\s+צבא)/i.test(query);

  let hits = av.items;
  let filterLabel = "כל המטוסים";

  if (wantAwacs) {
    hits = av.items.filter((i) => i.awacsSuspect);
    filterLabel = "מועמדים ל-AWACS (heuristic)";
  } else if (wantTanker) {
    hits = av.items.filter((i) => i.tankerSuspect);
    filterLabel = "מועמדים לתדלוק (heuristic)";
  } else if (wantMilitary) {
    hits = av.items.filter((i) => i.isMilitary);
    filterLabel = "מטוסים צבאיים (heuristic)";
  }

  const lines = [
    `מקור: עולם חי / ADS-B (${av.regionLabel})`,
    `סה"כ ${av.count} מטוסים במעקב · ${av.militaryCount ?? 0} צבאיים · ${av.awacsCount ?? 0} AWACS? · ${av.tankerCount ?? 0} תדלוק?`,
    `${filterLabel}: ${hits.length}`,
  ];

  if (wantAwacs && hits.length === 0) {
    lines.push(
      "ANSWER (AWACS): 0 מטוסים מזוהים כ-AWACS במעקב עולם חי (זיהוי heuristic — לא כל AWACS משדר ADS-B).",
    );
  } else if (wantAwacs && hits.length > 0) {
    const top = hits[0];
    lines.push(
      `ANSWER (AWACS): ${hits.length} מטוס(ים) מסומנים כ-AWACS? · ${top.callsign || top.icao24} · ${top.milLabel || "heuristic"}`,
    );
  }

  lines.push(
    ...hits.slice(0, 8).map((a, i) => `${i + 1}. ${formatAviationSampleLine(a)}`),
    "הערה: זיהוי צבאי/AWACS מבוסס על ICAO hex, callsign ו-NATO — כמו שכבת «תעופה» בעולם חי. לא כל מטוס צבאי משדר ADS-B.",
  );

  return lines.join("\n");
};
