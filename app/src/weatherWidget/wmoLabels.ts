import type { WeatherIconKind } from "./types";

export const WMO_HE: Record<number, string> = {
  0: "שמיים בהירים",
  1: "בהיר ברובו",
  2: "מעונן חלקית",
  3: "מעונן",
  45: "ערפל",
  48: "ערפל קפוא",
  51: "טפטוף קל",
  53: "טפטוף",
  55: "טפטוף כבד",
  61: "גשם קל",
  63: "גשם",
  65: "גשם כבד",
  71: "שלג קל",
  73: "שלג",
  75: "שלג כבד",
  80: "ממטרים",
  95: "סופת רעמים",
};

export function wmoLabel(code: number | null | undefined): string {
  if (code == null) return "—";
  return WMO_HE[code] ?? `קוד ${code}`;
}

export function wmoIconKind(code: number | null | undefined): WeatherIconKind {
  if (code == null) return "partly-cloudy";
  if (code === 0) return "clear";
  if (code === 1 || code === 2) return "partly-cloudy";
  if (code === 3) return "cloudy";
  if (code === 45 || code === 48) return "fog";
  if (code >= 51 && code <= 55) return "drizzle";
  if (code >= 61 && code <= 65 || code === 80) return "rain";
  if (code >= 71 && code <= 75) return "snow";
  if (code === 95) return "thunder";
  return "partly-cloudy";
}
