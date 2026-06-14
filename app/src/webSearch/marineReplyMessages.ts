import { isMarineInfraQuery, isShipsQuery } from "./intents";
import type { SearchIntent, SearchSourceResult } from "./types";

const parseShipLiveCount = (text: string): { live: number; demo: number; breakdown: string | null } | null => {
  const answer = text.match(/ANSWER \(ships live\):\s*(\d+)/)?.[1];
  const liveLine = text.match(/דיווח AIS חי \+ עולם חי:\s*(\d+)/)?.[1];
  const demo = text.match(/סימוני מסלול \(הדגמה[^:]*:\s*(\d+)/)?.[1];
  const legacy = text.match(/ספינות בטווח:\s*(\d+)\s*\(([^)]+)\)/);
  if (answer != null) {
    return {
      live: Number(answer),
      demo: demo != null ? Number(demo) : 0,
      breakdown: liveLine != null ? `דיווח AIS חי + עולם חי: ${liveLine}` : null,
    };
  }
  if (legacy) {
    return { live: Number(legacy[1]), demo: 0, breakdown: `ספינות בטווח: ${legacy[1]} (${legacy[2]})` };
  }
  return null;
};

const parseRegionLine = (text: string): string | null => {
  const m = text.match(/^אזור:\s*(.+)$/m);
  return m?.[1]?.trim() ?? null;
};

const parseShipSamples = (text: string, max = 4): string[] =>
  text
    .split("\n")
    .filter((l) => /^\d+\.\s/.test(l.trim()))
    .slice(0, max)
    .map((l) => l.trim());

const parseInfraCountLine = (text: string): string | null => {
  const m = text.match(/תשתיות ימיות בטווח:\s*([^\n]+)/);
  return m?.[1]?.trim() ?? null;
};

const parseInfraSamples = (text: string, max = 4): string[] =>
  text
    .split("\n")
    .filter((l) => /^\d+\.\s/.test(l.trim()))
    .slice(0, max)
    .map((l) => l.trim());

const buildShipsReply = (query: string, source: SearchSourceResult): string | null => {
  const parsed = parseShipLiveCount(source.text);
  if (!parsed) return null;
  const region = parseRegionLine(source.text) ?? "האזור המבוקש";
  const samples = parseShipSamples(source.text);
  const intro =
    /כמה|how\s+many/i.test(query)
      ? `לפי נתוני AIS / עולם חי לגבי ${region}:`
      : `לפי נתוני AIS / עולם חי סביב ${region}:`;

  const lines = [
    intro,
    `ANSWER: ${parsed.live} אוניות עם דיווח AIS חי${parsed.demo ? ` (+ ${parsed.demo} סימוני מסלול הדגמה)` : ""}`,
    parsed.breakdown ?? `דיווח חי: ${parsed.live}`,
  ];

  if (samples.length) lines.push("דוגמאות:", ...samples.map((s) => `• ${s.replace(/^\d+\.\s*/, "")}`));

  if (/כמה|how\s+many/i.test(query) && parsed.live === 0 && parsed.demo > 0) {
    lines.push(
      "הערה: 0 אוניות עם AIS חי באזור — Digitraffic לא מכסה את תעלת סואץ/ים תיכון. סימוני המסלול הם הדגמה מ«עולם חי», לא ספירה של אוניות בזמן אמת.",
    );
  } else if (parsed.demo > 0 && parsed.live > 0) {
    lines.push(`הערה: ${parsed.demo} סימוני מסלול (הדגמה) בנוסף ל-${parsed.live} עם AIS חי.`);
  }

  const updated = source.text.match(/^עודכן:\s*(.+)$/m)?.[1];
  if (updated) lines.push(`עודכן: ${updated}`);

  lines.push("Sources: ספינות (AIS / עולם חי)");
  lines.push("מקור: Digitraffic + cache עולם חי.");
  return lines.join("\n");
};

const buildInfraReply = (query: string, source: SearchSourceResult): string | null => {
  const countPart = parseInfraCountLine(source.text);
  if (countPart == null) return null;
  const region = parseRegionLine(source.text)?.replace(/\s*\(OpenStreetMap[^)]*\)/i, "").trim() ?? "האזור";
  const samples = parseInfraSamples(source.text);
  const intro = /מצופ|buoy/i.test(query)
    ? `לפי OpenStreetMap (Overpass) לגבי ${region}:`
    : /מגדלור|lighthouse/i.test(query)
      ? `לפי OpenStreetMap (Overpass) — מגדלורים ותשתיות ימיות סביב ${region}:`
      : `לפי OpenStreetMap (Overpass) לגבי ${region}:`;

  const lines = [intro, `תשתיות ימיות בטווח: ${countPart}`];
  if (samples.length) lines.push("דוגמאות:", ...samples.map((s) => `• ${s.replace(/^\d+\.\s*/, "")}`));
  lines.push("הערה: נתונים סטטיים מ-OpenStreetMap — לא ספירת כלי שייט בתנועה (AIS).");
  lines.push("מקור: OpenStreetMap / Overpass.");
  return lines.join("\n");
};

/** Fixed Hebrew when live marine providers returned data — avoids LLM ignoring SEARCH BRIEF. */
export function buildMarineLiveReply(
  query: string,
  _intents: SearchIntent[],
  sources: SearchSourceResult[],
): string | null {
  const q = query.trim();
  if (!q) return null;

  const shipsSource = sources.find((s) => s.provider === "ais-ships" && s.ok && s.text.trim());
  const infraSource = sources.find((s) => s.provider === "osm-overpass-marine" && s.ok && s.text.trim());

  if (isShipsQuery(q) && !isMarineInfraQuery(q) && shipsSource) {
    return buildShipsReply(q, shipsSource);
  }
  if (isMarineInfraQuery(q) && !isShipsQuery(q) && infraSource) {
    return buildInfraReply(q, infraSource);
  }
  if (isMarineInfraQuery(q) && isShipsQuery(q)) {
    if (infraSource) return buildInfraReply(q, infraSource);
    if (shipsSource) return buildShipsReply(q, shipsSource);
  }
  return null;
}

