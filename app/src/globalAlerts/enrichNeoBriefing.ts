import { resolveCountry } from "../webSearch/providers/restCountries";
import { estimateDiameterKm } from "./neoLiveMetrics";
import { formatNeoEta } from "./neoEta";
import { formatPopulationHe, reverseRegionLabel } from "./reverseGeoRegion";
import type { GlobeAlertEvent } from "./types";

export type NeoPublicRisk = "low" | "moderate" | "high" | "critical";

export type EnrichedNeoBriefing = {
  impactRegion: string;
  country?: string;
  isPopulated: boolean;
  populationLabel?: string;
  publicRisk: NeoPublicRisk;
  riskLabel: string;
  riskDetail: string;
  headline: string;
  coordsLabel: string;
  etaLabel: string;
};

export function assessNeoPublicRisk(
  ev: GlobeAlertEvent,
  diameterKm?: number,
): { level: NeoPublicRisk; label: string; detail: string } {
  const caLd = ev.distLd ?? 99;
  const d = diameterKm ?? ev.diameterKm ?? 0;

  if (caLd > 5) {
    return {
      level: "low",
      label: "נמוך",
      detail: `מעבר במרחק ${caLd.toFixed(1)} LD — ללא פגיעה צפויה בכדור הארץ`,
    };
  }
  if (caLd > 1) {
    return {
      level: "moderate",
      label: "בינוני",
      detail: "התקרבות קרובה — ניטור NASA פעיל, מסלול מעבר בטוח",
    };
  }
  if (caLd > 0.2 || d > 0.1) {
    return {
      level: "high",
      label: "גבוה",
      detail: ev.isPha ? "PHA — קרבה חריגה, מעקב חירום" : "קרבה חריגה — מעקב חירום",
    };
  }
  return {
    level: "critical",
    label: "קריטי",
    detail: "קרבה קיצונית — פרוטוקול התרעה מוגבר",
  };
}

export async function enrichNeoBriefing(ev: GlobeAlertEvent): Promise<EnrichedNeoBriefing> {
  const diameterKm = estimateDiameterKm(ev);
  const region = await reverseRegionLabel(ev.lat, ev.lon);
  const risk = assessNeoPublicRisk(ev, diameterKm);

  let country: string | undefined;
  let populationLabel: string | undefined;
  let isPopulated = false;

  const countryPart = region.split(" · ").pop()?.trim();
  if (countryPart && countryPart.length > 1 && !countryPart.startsWith("מעל")) {
    try {
      const row = await resolveCountry(countryPart);
      if (row?.population) {
        country = row.name;
        populationLabel = formatPopulationHe(row.population);
        isPopulated = row.population > 100_000;
      }
    } catch {
      /* skip */
    }
  }

  const isOcean = region.includes("מעל") || region.includes("ים");
  if (isOcean) isPopulated = false;

  const etaLabel = formatNeoEta(ev.approachTime ?? ev.time);
  const headline = `${ev.location} · ${etaLabel}`;

  return {
    impactRegion: isOcean ? region : `מעל ${region}`,
    country,
    isPopulated,
    populationLabel,
    publicRisk: risk.level,
    riskLabel: risk.label,
    riskDetail: risk.detail,
    headline,
    coordsLabel: `${ev.lat.toFixed(1)}°, ${ev.lon.toFixed(1)}°`,
    etaLabel,
  };
}
