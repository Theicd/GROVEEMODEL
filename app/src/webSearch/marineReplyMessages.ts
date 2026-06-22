import { isMarineInfraQuery, isShipsQuery } from "./intents";
import type { SearchIntent, SearchSourceResult } from "./types";

export type ParsedShipCounts = {
  live: number;
  demo: number;
  region: string;
  updated?: string;
};

export const parseShipLiveCount = (text: string): ParsedShipCounts | null => {
  const answer =
    text.match(/ANSWER \(ships live\):\s*(\d+)/i)?.[1] ??
    text.match(/ANSWER:\s*(\d+)\s+אוניות/i)?.[1];
  const liveLine = text.match(/דיווח AIS חי \+ עולם חי:\s*(\d+)/)?.[1];
  const demo = text.match(/סימוני מסלול \(הדגמה[^:]*:\s*(\d+)/i)?.[1];
  const legacy = text.match(/ספינות בטווח:\s*(\d+)\s*\(([^)]+)\)/);
  const region = text.match(/^אזור:\s*(.+)$/m)?.[1]?.trim() ?? "האזור המבוקש";
  const updated = text.match(/^עודכן:\s*(.+)$/m)?.[1]?.trim();

  if (answer != null) {
    return {
      live: Number(answer),
      demo: demo != null ? Number(demo) : 0,
      region,
      updated,
    };
  }
  if (legacy) {
    return { live: Number(legacy[1]), demo: 0, region, updated };
  }
  if (liveLine != null) {
    return {
      live: Number(liveLine),
      demo: demo != null ? Number(demo) : 0,
      region,
      updated,
    };
  }
  return null;
};

const normalizeShipRegionLabel = (raw: string): string =>
  raw.replace(/\s*\(bbox\)/i, "").replace(/\s*\(OpenStreetMap[^)]*\)/i, "").trim();

const regionWithPrep = (region: string): string => {
  if (/^(?:תעלת|מפרץ|נמל|מצר)/i.test(region)) return `ב${region}`;
  return `באזור ${region}`;
};

const parseLiveShipSamples = (text: string, max = 3): string[] =>
  text
    .split("\n")
    .filter((l) => /^\d+\.\s/.test(l.trim()) && !/הדגמה|מסלול \(הדגמה\)/i.test(l))
    .slice(0, max)
    .map((l) => `• ${l.trim().replace(/^\d+\.\s*/, "")}`);

/** User-facing canned reply — count-first, no demo markers or internal jargon. */
export const formatShipsCannedReply = (
  query: string,
  sourceText: string,
): string | null => {
  const parsed = parseShipLiveCount(sourceText);
  if (!parsed) return null;

  const region = normalizeShipRegionLabel(parsed.region);
  const place = regionWithPrep(region);
  const liveWord = parsed.live === 1 ? "אונייה" : "אוניות";
  const age = parsed.updated?.replace(/\s*UTC\s*$/i, "").trim();
  const ageBit = age ? ` · עדכון ${age}` : "";

  const lead = `${parsed.live} ${liveWord} ${place} לפי AIS${ageBit}.`;
  const lines = [lead];

  if (parsed.live > 0) {
    lines.push(...parseLiveShipSamples(sourceText));
  }

  lines.push("Sources: ספינות (AIS / עולם חי)");
  return lines.join("\n");
};

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

const buildShipsReply = (query: string, source: SearchSourceResult): string | null =>
  formatShipsCannedReply(query, source.text);

const buildInfraReply = (query: string, source: SearchSourceResult): string | null => {
  const countPart = parseInfraCountLine(source.text);
  if (countPart == null) return null;
  const region =
    parseShipLiveCount(source.text)?.region ??
    textRegion(source.text) ??
    "האזור";
  const samples = parseInfraSamples(source.text);
  const intro = /מצופ|buoy/i.test(query)
    ? `לפי OpenStreetMap (Overpass) לגבי ${region}:`
    : /מגדלור|lighthouse/i.test(query)
      ? `לפי OpenStreetMap (Overpass) — מגדלורים ותשתיות ימיות סביב ${region}:`
      : `לפי OpenStreetMap (Overpass) לגבי ${region}:`;

  const lines = [intro, `תשתיות ימיות בטווח: ${countPart}`];
  if (samples.length) lines.push("דוגמאות:", ...samples.map((s) => `• ${s.replace(/^\d+\.\s*/, "")}`));
  lines.push("Sources: OpenStreetMap / Overpass");
  return lines.join("\n");
};

const textRegion = (text: string): string | null => {
  const m = text.match(/^אזור:\s*(.+)$/m);
  return m?.[1]?.replace(/\s*\(OpenStreetMap[^)]*\)/i, "").trim() ?? null;
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
