/** Parse Horizons text block ($$SOE … $$EOE) for observer ephemeris rows. */

export type HorizonsObserverRow = {
  timeLabel: string;
  raHours: number;
  decDeg: number;
  distAu: number;
  deldotKmS: number;
};

function parseRaHms(h: string, m: string, s: string): number {
  return Number(h) + Number(m) / 60 + Number(s) / 3600;
}

function parseDecDms(sign: string, d: string, m: string, s: string): number {
  const mag = Math.abs(Number(d)) + Number(m) / 60 + Number(s) / 3600;
  return sign === "-" ? -mag : mag;
}

export function extractHorizonsResultBlock(text: string): string {
  const soe = text.indexOf("$$SOE");
  const eoe = text.indexOf("$$EOE");
  if (soe < 0 || eoe < 0 || eoe <= soe) return "";
  return text.slice(soe + 5, eoe);
}

/** Observer table: date, RA (h m s), Dec (d m s), APmag, S-brt, delta (AU), deldot (km/s). */
const OBS_LINE_RE =
  /^(\d{4}-\w{3}-\d{1,2}\s+\d{1,2}:\d{2})\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([+-])(\d+)\s+(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([-\d.]+)/;

export function parseHorizonsObserverRows(result: string): HorizonsObserverRow[] {
  const block = extractHorizonsResultBlock(result);
  if (!block.trim()) return [];

  const rows: HorizonsObserverRow[] = [];
  for (const line of block.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("Date")) continue;

    const m = trimmed.match(OBS_LINE_RE);
    if (!m) continue;

    rows.push({
      timeLabel: m[1],
      raHours: parseRaHms(m[2], m[3], m[4]),
      decDeg: parseDecDms(m[5], m[6], m[7], m[8]),
      distAu: Number.parseFloat(m[9]),
      deldotKmS: Number.parseFloat(m[10]),
    });
  }
  return rows;
}
