import type { SearchSourceResult } from "./types";

const localDateStr = (d = new Date()): string => {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
};

const daysBetween = (older: string, newer: string): number => {
  const a = new Date(`${older}T12:00:00`);
  const b = new Date(`${newer}T12:00:00`);
  if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime())) return 0;
  return Math.round((b.getTime() - a.getTime()) / 86_400_000);
};

/** Hebrew DATA AGE line for a single provider result (FX / market). */
export const formatDataAgeForSource = (source: SearchSourceResult): string | null => {
  if (!source.ok || !source.text.trim()) return null;
  const today = localDateStr();

  if (source.provider === "frankfurter-fx") {
    const m = source.text.match(/תאריך:\s*(\d{4}-\d{2}-\d{2})/);
    if (!m) return null;
    const days = daysBetween(m[1], today);
    if (days <= 0) return null;
    const weekend = [0, 6].includes(new Date().getDay());
    return `DATA AGE: שער ECB מ-${m[1]} — לא intraday${weekend ? '; בסופ"ש אין עדכון' : ""}`;
  }

  if (source.provider === "yahoo-finance") {
    const m = source.text.match(/עדכון \(Yahoo Finance\):\s*(\d{4}-\d{2}-\d{2})/);
    if (!m) return null;
    const days = daysBetween(m[1], today);
    if (days <= 0) return null;
    return `DATA AGE: סגירת מסחר ${m[1]} — השוק סגור; לא מחיר חי`;
  }

  return null;
};

/** DATA AGE lines for search brief header (from all sources). */
export const buildDataAgeLines = (sources: SearchSourceResult[]): string[] => {
  const out: string[] = [];
  for (const s of sources) {
    const line = formatDataAgeForSource(s);
    if (line && !out.includes(line)) out.push(line);
  }
  return out;
};

export const hasStaleDataAge = (ctx: string): boolean => /DATA AGE:/i.test(ctx);

export const queryAsksLiveNow = (prompt: string): boolean => /כרגע|עכשיו|now|today|היום/i.test(prompt);
