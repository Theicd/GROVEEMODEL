export type GlobeAlertEventType =
  | "earthquake"
  | "tsunami"
  | "hurricane"
  | "tornado"
  | "fire"
  | "flood"
  | "volcano"
  | "disaster"
  | "neo"
  | "fireball";

/** USGS earthquakes: last 10 minutes only (a quake is instant, not ongoing). */
export const EQ_LIVE_WINDOW_MS = 10 * 60 * 1000;

/** Fallback recency for non-ongoing GDACS types. */
export const EARTH_LIVE_WINDOW_MS = 60 * 60 * 1000;

/** Global alerts: show all magnitudes from the live hour feed (filtered by age in alertFilters). */
export const GLOBAL_ALERTS_EQ_MIN_MAG = 0;

export type GlobeAlertEvent = {
  id: string;
  type: GlobeAlertEventType;
  lat: number;
  lon: number;
  location: string;
  time: number;
  source: "usgs" | "gdacs" | "nasa-jpl";
  magnitude?: number;
  depth?: number;
  category?: number;
  alertLevel?: string;
  severity?: number;
  /** GDACS tropical cyclone identifiers for track fetch. */
  gdacsEventId?: number;
  gdacsEpisodeId?: number;
  reportUrl?: string;
  severityText?: string;
  /** NEO / fireball (NASA JPL) */
  designation?: string;
  distAu?: number;
  distLd?: number;
  distMinLd?: number;
  distMaxLd?: number;
  vRel?: number;
  vInf?: number;
  approachTime?: number;
  approachLabel?: string;
  hMagnitude?: number;
  diameterKm?: number;
  isPha?: boolean;
  scoutNeoScore?: number;
  /** GDACS episode still marked current on gdacs.org */
  gdacsIsCurrent?: boolean;
  /** GDACS episode end (todate) */
  gdacsEndTime?: number;
  /** Last GDACS update */
  updatedTime?: number;
  /** CAD listed before Horizons track is ready — sidebar only, no globe marker yet. */
  trackPending?: boolean;
  impactKt?: number;
  velocityKmS?: number;
  altitudeKm?: number;
};

export const EVENT_TYPE_LABELS: Record<
  GlobeAlertEventType,
  { label: string; color: string; hex: number }
> = {
  earthquake: { label: "רעידת אדמה", color: "#FF4444", hex: 0xff4444 },
  tsunami: { label: "צונאמי", color: "#00BBFF", hex: 0x00bbff },
  tornado: { label: "טורנדו", color: "#BBBBBB", hex: 0xbbbbbb },
  hurricane: { label: "הוריקן / סופה", color: "#AA66FF", hex: 0xaa66ff },
  fire: { label: "שריפה", color: "#FF8800", hex: 0xff8800 },
  flood: { label: "שיטפון", color: "#4488FF", hex: 0x4488ff },
  volcano: { label: "הר געש", color: "#FF5500", hex: 0xff5500 },
  disaster: { label: "אסון טבע", color: "#CCAA44", hex: 0xccaa44 },
  neo: { label: "אסטרואיד קרוב", color: "#66DDFF", hex: 0x66ddff },
  fireball: { label: "בוליד / Fireball", color: "#FFAA33", hex: 0xffaa33 },
};
