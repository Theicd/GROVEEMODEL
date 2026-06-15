import { isCrossSourceQuery } from "./crossSourceIntents";
import type { SearchIntent, SearchSourceResult } from "./types";

export type CrossSourceMetrics = {
  regionLabel?: string;
  weather?: {
    place?: string;
    condition?: string;
    windKmh?: number;
    stormLike: boolean;
  };
  aviation?: { count: number; region?: string };
  ships?: { count: number; region?: string };
  airQuality?: { aqi?: number; pm25?: number };
};

const STORM_TEXT_RE = /סופ|רעמ|גשם\s*כבד|ממטר|storm|thunder|hurricane|typhoon|טיפונ/i;

const parseIntSafe = (raw: string | undefined): number | undefined => {
  if (raw == null) return undefined;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) ? n : undefined;
};

const parseFloatSafe = (raw: string | undefined): number | undefined => {
  if (raw == null) return undefined;
  const n = parseFloat(raw);
  return Number.isFinite(n) ? n : undefined;
};

/** Pull comparable numbers from live provider text for cross-source synthesis. */
export const extractCrossSourceMetrics = (
  sources: SearchSourceResult[],
  regionLabel?: string,
): CrossSourceMetrics => {
  const metrics: CrossSourceMetrics = { regionLabel };

  const weather = sources.find((s) => s.provider === "open-meteo" && s.ok && s.text.trim());
  if (weather) {
    const lines = weather.text.split("\n");
    const condition = lines.find((l) => l.startsWith("מצב:"))?.replace("מצב:", "").trim();
    const place = lines.find((l) => l.startsWith("מיקום:"))?.replace("מיקום:", "").trim();
    const windKmh = parseFloatSafe(lines.find((l) => /^רוח:/i.test(l))?.match(/([\d.]+)\s*km/i)?.[1]);
    metrics.weather = {
      place,
      condition,
      windKmh,
      stormLike: STORM_TEXT_RE.test(weather.text) || (windKmh != null && windKmh >= 45),
    };
  }

  const aviation = sources.find((s) => s.provider === "adsb-aviation" && s.ok && s.text.trim());
  if (aviation) {
    const count = parseIntSafe(
      aviation.text.match(/מטוסים (?:בטווח|באוויר)[^:\n]*:\s*(\d+)/i)?.[1] ??
        aviation.text.match(/סה[״"']?כ\s+(\d+)\s+מטוסים/i)?.[1] ??
        aviation.text.match(/כל\s+המטוסים:\s*(\d+)/i)?.[1],
    );
    if (count != null) {
      metrics.aviation = {
        count,
        region: aviation.text.match(/^אזור:\s*(.+)$/m)?.[1]?.trim(),
      };
    }
  }

  const ships = sources.find((s) => s.provider === "ais-ships" && s.ok && s.text.trim());
  if (ships) {
    const count = parseIntSafe(
      ships.text.match(/ספינות בטווח:\s*(\d+)/i)?.[1] ??
        ships.text.match(/ANSWER \(ships live\):\s*(\d+)/i)?.[1],
    );
    if (count != null) metrics.ships = { count };
  }

  const aq = sources.find((s) => s.provider === "open-meteo-air-quality" && s.ok && s.text.trim());
  if (aq) {
    metrics.airQuality = {
      aqi: parseIntSafe(aq.text.match(/US AQI:\s*(\d+)/i)?.[1]),
      pm25: parseFloatSafe(aq.text.match(/PM2\.5:\s*([\d.]+)/i)?.[1]),
    };
  }

  return metrics;
};

export const shouldBuildCrossSourceCorrelation = (
  query: string,
  intents: SearchIntent[],
): boolean => isCrossSourceQuery(query) || intents.filter((i) => i !== "wikipedia").length >= 2;

/** Rule-based synthesis lines for the model brief (Phase 5). */
export const buildCrossSourceCorrelationLines = (
  query: string,
  metrics: CrossSourceMetrics,
  intents: SearchIntent[],
): string[] => {
  if (intents.length < 2) return [];

  const region =
    metrics.regionLabel ??
    metrics.weather?.place ??
    metrics.aviation?.region ??
    "האזור המשותף";

  const lines: string[] = [];
  const yesNo = /^האם\s+/i.test(query) || /is there|yes or no/i.test(query);

  if (metrics.weather && metrics.aviation) {
    const { stormLike, condition, windKmh } = metrics.weather;
    const planes = metrics.aviation.count;
    if (yesNo || /סופה.*מטוס|מטוס.*סופה|מזג.*מטוס|תעופ.*מזג|חריג.*(?:מזג|תנועה|תעופ)/i.test(query)) {
      if (stormLike && planes > 0) {
        lines.push(
          `CORRELATION: כן — ${region}: מז"א מצביע על מזג קשה (${condition ?? "סוער"}${windKmh ? `, רוח ~${windKmh} km/h` : ""}) + ${planes} מטוסים ב-ADS-B.`,
        );
      } else if (stormLike && planes === 0) {
        lines.push(
          `CORRELATION: מז"א מצביע על מזג קשה (${condition ?? "סוער"}) אך 0 מטוסים בטווח ADS-B — ייתכן כיסוי חלקי.`,
        );
      } else if (!stormLike && planes > 0) {
        lines.push(
          `CORRELATION: לא סופה ברורה — מז"א: ${condition ?? "רגוע"}; ${planes} מטוסים פעילים ב-${metrics.aviation.region ?? region}.`,
        );
      } else {
        lines.push(`CORRELATION: ${region} — מז"א: ${condition ?? "—"}; מטוסים: ${planes}.`);
      }
    } else {
      lines.push(
        `CORRELATION GEO: ${region} — מז"א (${condition ?? "—"}) + ADS-B (${planes} מטוסים).`,
      );
    }
  }

  if (metrics.weather && metrics.ships && !lines.length) {
    lines.push(
      `CORRELATION GEO: ${region} — מז"א: ${metrics.weather.condition ?? "—"}; ספינות AIS: ${metrics.ships.count}.`,
    );
  }

  if (metrics.airQuality?.aqi != null && metrics.weather && !lines.length) {
    lines.push(
      `CORRELATION: ${region} — US AQI ${metrics.airQuality.aqi}, מז"א: ${metrics.weather.condition ?? "—"}.`,
    );
  }

  if (!lines.length) {
    const parts: string[] = [];
    if (metrics.weather) parts.push(`מז"א: ${metrics.weather.condition ?? "יש נתונים"}`);
    if (metrics.aviation) parts.push(`מטוסים: ${metrics.aviation.count}`);
    if (metrics.ships) parts.push(`ספינות: ${metrics.ships.count}`);
    if (metrics.airQuality?.aqi != null) parts.push(`AQI: ${metrics.airQuality.aqi}`);
    if (parts.length >= 2) {
      lines.push(`CORRELATION GEO (${region}): ${parts.join(" · ")}`);
    }
  }

  return lines;
};
