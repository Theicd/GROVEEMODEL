/** Shared military / AWACS heuristics — mirrors public/reality/israel.js `_classifyAircraft`. */

export type AircraftClassification = {
  mil: boolean;
  label: string;
  awacsSuspect: boolean;
};

const MIL_HEX: Array<[string, string, string]> = [
  ["ae", "af", 'צבא ארה"ב'],
  ["43c", "43c", "UK Military"],
  ["3b", "3b", "French Military"],
  ["3e", "3e", "German Military"],
  ["150", "15f", "Russian Military"],
  ["4b8", "4bf", "Turkish Military"],
  ["730", "737", "Iranian Military"],
  ["778", "77f", "Syrian Military"],
  ["e4", "e4", "NATO"],
];

const IL_AIRLINE_CS = ["ELY", "LY", "ICL", "6H", "ISR", "5C", "RVR"];

const AWACS_CALLSIGN =
  /\b(?:AWACS|SENTRY\d*|SNTRY|NATO\d*|E-?3[A-Z]?|CON\d+|DRAGON|IRON|MAGIC|SINCE|DUKE|Gulf\s*Stream)/i;
const AWACS_LABEL = /NATO|AWACS|E-?3|Sentry|מודיעין/i;
const TANKER_CALLSIGN = /\b(?:TANK|TANKER|NCHO|NCH\d|MOOSE|GUARD|METRO|GOLD|RCH\d|DUKE\d)\b/i;

export const isAwacsSuspect = (
  callsign?: string | null,
  milLabel?: string | null,
  category?: string | number | null,
): boolean => {
  const cs = (callsign ?? "").trim();
  const lbl = milLabel ?? "";
  if (AWACS_CALLSIGN.test(cs) || AWACS_LABEL.test(lbl)) return true;
  const cat = String(category ?? "");
  if (/E-?3|AWACS|Sentry/i.test(cat)) return true;
  return false;
};

export const isTankerSuspect = (callsign?: string | null, category?: string | number | null): boolean => {
  const cs = (callsign ?? "").trim();
  if (TANKER_CALLSIGN.test(cs)) return true;
  return /tanker|KC-?1[035678]|A332|MRTT/i.test(String(category ?? ""));
};

export const classifyAircraft = (
  icao24?: string | null,
  callsign?: string | null,
  country?: string | null,
  category?: string | number | null,
  dbFlags?: number,
): AircraftClassification => {
  const hex = (icao24 ?? "").toLowerCase();
  for (const [lo, hi, label] of MIL_HEX) {
    if (hex >= lo && hex <= `${hi}ffff`) {
      const awacsSuspect = isAwacsSuspect(callsign, label, category);
      return { mil: true, label, awacsSuspect };
    }
  }

  const cs = (callsign ?? "").trim().toUpperCase();
  if (!cs && country && country !== "Israel") {
    const label = `${country} (ללא callsign)`;
    return { mil: true, label, awacsSuspect: isAwacsSuspect(callsign, label, category) };
  }

  const catNum = Number(category) || 0;
  if (catNum === 14) {
    const label = `UAV ${country ?? ""}`.trim();
    return { mil: true, label, awacsSuspect: false };
  }

  if (IL_AIRLINE_CS.some((p) => cs.startsWith(p))) {
    return { mil: false, label: "", awacsSuspect: false };
  }

  if ((dbFlags ?? 0) & 1) {
    const label = `MIL ${country ?? ""}`.trim();
    return { mil: true, label, awacsSuspect: isAwacsSuspect(callsign, label, category) };
  }

  const awacsSuspect = isAwacsSuspect(callsign, "", category);
  return { mil: awacsSuspect, label: awacsSuspect ? "AWACS (callsign)" : "", awacsSuspect };
};
