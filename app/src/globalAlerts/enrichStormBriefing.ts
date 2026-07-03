import { resolveCountry } from "../webSearch/providers/restCountries";
import { parseWindKmh } from "./hurricaneIntensity";
import type { StormBriefing } from "./fetchStormBriefing";
import { bearingToCompassHe } from "./parseStormGeometry";
import { formatPopulationHe, reverseRegionLabel } from "./reverseGeoRegion";
import type { GlobeAlertEvent } from "./types";

export type EnrichedStormBriefing = {
  briefing: StormBriefing;
  currentRegion: string;
  targetRegion?: string;
  movementLine: string;
  headline: string;
  narrativeLines: string[];
  populationAtRisk?: number;
  populationLabel?: string;
  windKmh?: number;
  etaHours?: number;
  bearingDeg?: number;
  bearingLabel?: string;
  trackSpeedKmh?: number;
  coordsLabel?: string;
};

function hoursUntil(ts?: number): number | undefined {
  if (!ts) return undefined;
  const h = (ts - Date.now()) / 3600000;
  return h > 0 ? Math.round(h) : undefined;
}

export async function enrichStormBriefing(
  briefing: StormBriefing,
  ev: GlobeAlertEvent,
): Promise<EnrichedStormBriefing> {
  const { track, currentPos, forecastTarget, affectedCountries, gdacsCountry, gdacsCountryOnLand } =
    briefing;

  const [currentRegionRaw, targetRegionRaw] = await Promise.all([
    reverseRegionLabel(currentPos.lat, currentPos.lon),
    forecastTarget
      ? reverseRegionLabel(forecastTarget.lat, forecastTarget.lon)
      : Promise.resolve(undefined),
  ]);

  const currentRegion =
    gdacsCountryOnLand ||
    gdacsCountry ||
    (affectedCountries[0]?.name ? `ליד ${affectedCountries[0].name}` : currentRegionRaw);

  const targetRegion = targetRegionRaw;
  const windKmh = parseWindKmh(ev.severityText);
  const bearing = track.bearingDeg != null ? bearingToCompassHe(track.bearingDeg) : null;
  const speed = track.speedKmh != null ? Math.round(track.speedKmh) : null;
  const etaHours = hoursUntil(forecastTarget?.time);

  const countryNames = new Set<string>();
  for (const c of affectedCountries) countryNames.add(c.name);
  if (gdacsCountry) countryNames.add(gdacsCountry);
  if (gdacsCountryOnLand) countryNames.add(gdacsCountryOnLand);

  let populationAtRisk = 0;
  const popParts: string[] = [];
  for (const name of countryNames) {
    if (!name || name.length < 2) continue;
    try {
      const row = await resolveCountry(name);
      if (row?.population) {
        populationAtRisk += row.population;
        popParts.push(row.name);
      }
    } catch {
      /* skip */
    }
  }

  const movementParts: string[] = [];
  if (bearing) movementParts.push(`נוסעת ${bearing}`);
  if (speed) movementParts.push(`${speed} קמ"ש`);
  if (targetRegion) movementParts.push(`לכיוון ${targetRegion.split(" · ")[0]}`);
  const movementLine = movementParts.join(" · ") || "מסלול בתהליך עדכון";

  const coordsLabel = `${currentPos.lat.toFixed(1)}°${currentPos.lat >= 0 ? "N" : "S"}, ${Math.abs(currentPos.lon).toFixed(1)}°${currentPos.lon >= 0 ? "E" : "W"}`;

  const cat = ev.category ?? "?";
  const windPart = windKmh ? ` · ${windKmh} קמ"ש` : "";
  const headline = `${ev.location} · קט ${cat}${windPart}`;

  const narrativeLines: string[] = [];
  narrativeLines.push(`📡 מיקום חי: ${currentRegion}`);
  narrativeLines.push(
    `🌐 קואורדינטות: ${currentPos.lat.toFixed(1)}°${currentPos.lat >= 0 ? "N" : "S"}, ${Math.abs(currentPos.lon).toFixed(1)}°${currentPos.lon >= 0 ? "E" : "W"}`,
  );
  narrativeLines.push(`🧭 ${movementLine}`);
  if (targetRegion && targetRegion !== currentRegion) {
    narrativeLines.push(`🎯 יעד תחזית GDACS: ${targetRegion}`);
  }
  if (etaHours != null && etaHours > 0) {
    narrativeLines.push(`⏱ הגעה משוערת לנקודת קצה תחזית: ~${etaHours} שעות`);
  }
  if (populationAtRisk > 0) {
    narrativeLines.push(
      `👥 אוכלוסיה באזורי סיכון (${popParts.slice(0, 3).join(", ")}): ${formatPopulationHe(populationAtRisk)}`,
    );
  } else if (affectedCountries.length) {
    narrativeLines.push(
      `⚠️ מדינות מושפעות (GDACS): ${affectedCountries.map((c) => c.name).join(" · ")}`,
    );
  }
  if (ev.severityText) {
    narrativeLines.push(`💨 ${ev.severityText}`);
  }
  narrativeLines.push(`📊 מסלול: ${track.observed.length} נק' · תחזית: ${track.forecast.length} נק'`);

  return {
    briefing,
    currentRegion,
    targetRegion,
    movementLine,
    headline,
    narrativeLines,
    populationAtRisk: populationAtRisk || undefined,
    populationLabel: populationAtRisk ? formatPopulationHe(populationAtRisk) : undefined,
    windKmh,
    etaHours,
    bearingDeg: track.bearingDeg,
    bearingLabel: bearing ?? undefined,
    trackSpeedKmh: speed ?? undefined,
    coordsLabel,
  };
}
