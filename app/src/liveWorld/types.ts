import type { SearchSourceResult } from "../webSearch/types";

export type LiveEarthquakeItem = {
  magnitude: number | null;
  place: string;
  time: number;
  lat?: number;
  lon?: number;
  url?: string;
  tsunami?: boolean;
};

export type LiveDisasterItem = {
  eventName: string;
  country: string;
  alertLevel: string;
  eventType?: string;
  url?: string;
};

export type LiveIssPosition = {
  lat: number;
  lon: number;
  altitudeKm: number;
  velocityKmh?: number;
};

export type LiveShipItem = {
  name: string;
  lat: number;
  lon: number;
  speedKn?: number;
  destination?: string;
  source: string;
};

export type LiveAviationItem = {
  icao24?: string;
  callsign?: string;
  country?: string;
  lat?: number;
  lon?: number;
  alt?: number;
  isMilitary?: boolean;
  milLabel?: string;
  awacsSuspect?: boolean;
  tankerSuspect?: boolean;
};

export type LiveAviationSummary = {
  count: number;
  militaryCount?: number;
  awacsCount?: number;
  tankerCount?: number;
  regionLabel: string;
  sample: string[];
  items?: LiveAviationItem[];
};

/** Cached snapshot from Reality Live + direct fetches. */
export type LiveWorldSnapshot = {
  fetchedAt: number;
  source: "globe" | "fetch" | "mixed";
  earthquake?: { items: LiveEarthquakeItem[]; feedLabel: string };
  disasters?: { items: LiveDisasterItem[]; feedLabel: string; fetchedAt: number };
  iss?: LiveIssPosition;
  ships?: { regionLabel: string; count: number; items: LiveShipItem[] };
  marineInfra?: {
    regionLabel: string;
    items: Array<{ name: string; kind: string; lat?: number; lon?: number }>;
  };
  aviation?: LiveAviationSummary;
};

export type LiveWorldLayer = keyof Pick<
  LiveWorldSnapshot,
  "earthquake" | "disasters" | "iss" | "ships" | "marineInfra" | "aviation"
>;

export type SnapshotSearchFallback = SearchSourceResult;
