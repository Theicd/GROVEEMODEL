import {
  buildLiveDisastersHitsFromSnapshot,
  disasterItemToHit,
  earthquakeItemToHit,
} from "../searchResults/liveDisastersHits";
import type { LiveWorldSnapshot } from "../liveWorld/types";
import type { UnifiedSearchHit } from "../searchResults/types";
import type { GlobeAlertEvent, GlobeAlertEventType } from "./types";
import { isGdacsEventLive } from "../realityData/providers/disasters";
import { EQ_LIVE_WINDOW_MS, GLOBAL_ALERTS_EQ_MIN_MAG } from "./types";
import { parseWindKmh, windToCategory } from "./hurricaneIntensity";

const GDACS_TYPE_MAP: Record<string, GlobeAlertEventType> = {
  TC: "hurricane",
  FL: "flood",
  WF: "fire",
  TS: "tsunami",
  VO: "volcano",
};

const alertToCategory = (alert?: string): number => {
  if (/red/i.test(alert ?? "")) return 4;
  if (/orange/i.test(alert ?? "")) return 3;
  return 2;
};

const alertToSeverity = (alert?: string): number => {
  if (/red/i.test(alert ?? "")) return 2;
  if (/orange/i.test(alert ?? "")) return 1.5;
  return 1;
};

export const hitToGlobeEvent = (hit: UnifiedSearchHit): GlobeAlertEvent | null => {
  const lat = hit.meta?.lat;
  const lon = hit.meta?.lon;
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  if (hit.kind === "earthquake") {
    const mag = hit.meta?.magnitude ?? 0;
    if (mag < GLOBAL_ALERTS_EQ_MIN_MAG) return null;
    return {
      id: hit.id,
      type: hit.meta?.engine?.includes("Tsunami") ? "tsunami" : "earthquake",
      lat: lat!,
      lon: lon!,
      location: hit.title.replace(/^M[\d.?]+\s*·\s*/, ""),
      time: hit.publishedTs ?? Date.now(),
      source: "usgs",
      magnitude: hit.meta?.magnitude,
      depth: hit.meta?.depth,
    };
  }

  if (hit.kind === "disaster") {
    const dt = hit.meta?.disasterType?.toUpperCase() ?? "";
    if (dt === "EQ") return null;
    const type = GDACS_TYPE_MAP[dt] ?? "disaster";
    const alert = hit.meta?.alertLevel;
    let category = type === "hurricane" ? alertToCategory(alert) : undefined;
    if (type === "hurricane") {
      const wind = parseWindKmh(hit.meta?.severityText);
      if (wind != null) category = Math.max(category ?? 2, windToCategory(wind));
    }
    return {
      id: hit.id,
      type,
      lat: lat!,
      lon: lon!,
      location: hit.title,
      time: hit.publishedTs ?? Date.now(),
      source: "gdacs",
      category: type === "hurricane" ? category : undefined,
      alertLevel: alert,
      severity: alertToSeverity(alert),
      gdacsEventId: hit.meta?.gdacsEventId,
      gdacsEpisodeId: hit.meta?.gdacsEpisodeId,
      reportUrl: hit.url,
      severityText: hit.meta?.severityText,
      gdacsIsCurrent: hit.meta?.gdacsIsCurrent,
      gdacsEndTime: hit.meta?.gdacsEndTime,
      updatedTime: hit.meta?.dateModified ?? hit.publishedTs,
    };
  }

  return null;
};

export const hitsToGlobeEvents = (hits: UnifiedSearchHit[]): GlobeAlertEvent[] => {
  const out: GlobeAlertEvent[] = [];
  const seen = new Set<string>();
  for (const hit of hits) {
    const ev = hitToGlobeEvent(hit);
    if (!ev || seen.has(ev.id)) continue;
    seen.add(ev.id);
    out.push(ev);
  }
  return out.sort((a, b) => b.time - a.time);
};

export const snapshotToGlobeEvents = (snapshot: LiveWorldSnapshot | null): GlobeAlertEvent[] => {
  if (!snapshot) return [];
  const hits = buildLiveDisastersHitsFromSnapshot(snapshot);
  return hitsToGlobeEvents(hits);
};

export const rawSnapshotToGlobeEvents = (snapshot: LiveWorldSnapshot): GlobeAlertEvent[] => {
  const hits: UnifiedSearchHit[] = [];
  const eqLabel = snapshot.earthquake?.feedLabel ?? "USGS";
  const now = Date.now();
  for (const [i, item] of (snapshot.earthquake?.items ?? []).entries()) {
    if (item.lat == null || item.lon == null) continue;
    if ((item.magnitude ?? 0) < GLOBAL_ALERTS_EQ_MIN_MAG) continue;
    if (now - item.time > EQ_LIVE_WINDOW_MS) continue;
    hits.push(earthquakeItemToHit(item, i, eqLabel));
  }
  const gdacsLabel = snapshot.disasters?.feedLabel ?? "GDACS";
  for (const [i, item] of (snapshot.disasters?.items ?? []).entries()) {
    if (item.lat == null || item.lon == null) continue;
    if (!isGdacsEventLive(item)) continue;
    hits.push(disasterItemToHit(item, i, gdacsLabel));
  }
  return hitsToGlobeEvents(hits);
};
