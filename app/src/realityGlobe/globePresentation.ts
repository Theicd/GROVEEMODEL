import { normalizeCountrySearchName } from "../webSearch/queryExtract";

const DISPLAY_NAMES: Record<string, string> = {
  Germany: "גרמניה",
  France: "צרפת",
  Israel: "ישראל",
  "United States": "ארצות הברית",
  "United Kingdom": "בריטניה",
  Spain: "ספרד",
  Italy: "איטליה",
  Japan: "יפן",
  China: "סין",
  Brazil: "ברזיל",
  Canada: "קנדה",
  Mexico: "מקסיקו",
  Russia: "רוסיה",
  Australia: "אוסטרליה",
  Egypt: "מצרים",
  Turkey: "טורקיה",
  Greece: "יוון",
  Jordan: "ירדן",
  Lebanon: "לבנון",
  Poland: "פולין",
  Netherlands: "הולנד",
  Belgium: "בלגיה",
  Sweden: "שבדיה",
  Norway: "נורווגיה",
  Finland: "פינלנד",
};

export function placeDisplayNameHe(raw: string): string {
  const norm = normalizeCountrySearchName(raw);
  return DISPLAY_NAMES[norm] || raw.trim();
}

export function buildGlobePlaceReply(placeName: string): string {
  const label = placeDisplayNameHe(placeName);
  return `הצגתי את ${label} על המפה בפאנל REALITY LIVE מימין.`;
}

export function buildPlacesMapReply(placeLabel: string, osmUrl?: string): string {
  const label = placeDisplayNameHe(placeLabel);
  const link = osmUrl ? `\nOpenStreetMap: ${osmUrl}` : "";
  return `תחנת/מיקום: ${label}. המפה נפתחה מימין עם סימון OpenStreetMap.${link}\nSources: OpenStreetMap (Nominatim)`;
}

export function buildRouteMapReply(from: string, to: string, km?: string, driveTime?: string): string {
  const parts = [
    `מסלול: ${from} → ${to}.`,
    km ? `מרחק: ${km} ק"מ.` : "",
    driveTime ? `זמן נסיעה משוער: ${driveTime}.` : "",
    "המסלול מוצג על המפה מימין (OSRM + OpenStreetMap).",
    "Sources: OSRM · OpenStreetMap",
  ].filter(Boolean);
  return parts.join("\n");
}

export const GLOBE_PRESENTATION_APPEND = `REALITY LIVE MAP (mandatory when user asked to show a place on the map):
The app opened an interactive Cesium map panel beside the chat and focused on the requested location.
You CAN show places on the map — do NOT say you only generate text or cannot show maps.
Confirm briefly in Hebrew that the map is open and focused on the place the user asked for.
Do not claim you opened it if the user did not ask for a map — this note applies only when map context is provided.`;
