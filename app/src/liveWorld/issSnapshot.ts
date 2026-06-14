import type { SearchSourceResult } from "../webSearch/types";
import { getCachedLiveWorldSnapshot } from "./snapshotStore";
import type { LiveWorldSnapshot } from "./types";

/** ISS moves fast but globe refresh ~45s — keep cache usable for chat fallback. */
export const LIVE_WORLD_ISS_MAX_AGE_MS = 600_000;

export const formatIssSnapshotText = (snap: LiveWorldSnapshot): string | null => {
  const iss = snap.iss;
  if (!iss) return null;
  const ageSec = Math.round((Date.now() - snap.fetchedAt) / 1000);
  const ageNote = ageSec < 120 ? "עדכני" : `גיל cache: ~${Math.round(ageSec / 60)} דק'`;
  return [
    "מיקום ISS (עולם חי / wheretheiss):",
    `קו רוחב: ${iss.lat.toFixed(2)}°`,
    `קו אורך: ${iss.lon.toFixed(2)}°`,
    `גובה: ${iss.altitudeKm.toFixed(0)} km`,
    iss.velocityKmh != null ? `מהירות: ${iss.velocityKmh.toFixed(0)} km/h` : "",
    `ANSWER (ISS position): ${iss.lat.toFixed(2)}°, ${iss.lon.toFixed(2)}° · ${iss.altitudeKm.toFixed(0)} km`,
    `עודכן: ${new Date(snap.fetchedAt).toISOString().replace("T", " ").slice(0, 19)} UTC (${ageNote})`,
    "מקור: cache עולם חי (REALITY LIVE).",
  ]
    .filter(Boolean)
    .join("\n");
};

export const issSearchResultFromLiveWorld = (
  query: string,
  maxAgeMs = LIVE_WORLD_ISS_MAX_AGE_MS,
): SearchSourceResult | null => {
  if (!/\biss\b|תחנת\s+(?:ה)?חלל|space\s+station|החלל\s+הבינלאומ|היכן.*חלל/i.test(query)) {
    return null;
  }
  const snap = getCachedLiveWorldSnapshot(maxAgeMs);
  if (!snap?.iss) return null;
  const text = formatIssSnapshotText(snap);
  if (!text) return null;
  return {
    provider: "iss-tracker",
    label: "תחנת חלל (עולם חי / ISS)",
    ok: true,
    text,
    url: "https://api.wheretheiss.at",
    latencyMs: 0,
  };
};

export const isIssLikeQuery = (query: string): boolean =>
  /\biss\b|תחנת\s+(?:ה)?חלל|space\s+station|החלל\s+הבינלאומ|היכן.*(?:תחנת\s+)?(?:ה)?חלל|where.*space\s+station/i.test(
    query,
  );
