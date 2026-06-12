import { geocodePlace, formatPlaceLabel } from "../geoResolve";
import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractLocationPhrase, extractTimeZonePair } from "../queryExtract";

type TimeApiZone = {
  timeZone?: string;
  currentLocalTime?: string;
  currentUtcOffset?: { seconds?: number };
  standardUtcOffset?: { seconds?: number };
  hasDayLightSaving?: boolean;
  isDayLightSavingActive?: boolean;
};

const DAY_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

const fetchTimezone = async (timezone: string): Promise<TimeApiZone> =>
  fetchJson<TimeApiZone>(
    `https://timeapi.io/api/TimeZone/zone?timeZone=${encodeURIComponent(timezone)}`,
  );

const formatOffset = (seconds: number | undefined): string => {
  if (seconds == null) return "—";
  const hours = seconds / 3600;
  const sign = hours >= 0 ? "+" : "";
  return `UTC${sign}${hours}`;
};

const formatLocalTime = (iso: string | undefined): string => {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString("he-IL", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
};

export const fetchWorldTimeSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "world-time" as const;
  const label = "שעון עולמי (TimeAPI.io)";
  try {
    const pair = extractTimeZonePair(query);
    if (pair) {
      const [aName, bName] = pair;
      const placeA = await geocodePlace(aName);
      const placeB = await geocodePlace(bName);
      if (!placeA?.timezone || !placeB?.timezone) {
        return {
          provider,
          label,
          ok: false,
          text: "",
          error: "לא נמצאו אזורי זמן לשני המיקומים",
          latencyMs: Math.round(performance.now() - started),
        };
      }
      const [timeA, timeB] = await Promise.all([
        fetchTimezone(placeA.timezone),
        fetchTimezone(placeB.timezone),
      ]);
      const offA = timeA.currentUtcOffset?.seconds;
      const offB = timeB.currentUtcOffset?.seconds;
      const diff =
        offA != null && offB != null ? Math.round(((offB - offA) / 3600) * 10) / 10 : null;
      const lines = [
        `${formatPlaceLabel(placeA)} (${placeA.timezone}): ${formatLocalTime(timeA.currentLocalTime)} · ${formatOffset(offA)}`,
        `${formatPlaceLabel(placeB)} (${placeB.timezone}): ${formatLocalTime(timeB.currentLocalTime)} · ${formatOffset(offB)}`,
      ];
      if (diff != null) {
        lines.push(`הפרש: ${diff > 0 ? "+" : ""}${diff} שעות`);
      }
      return {
        provider,
        label,
        ok: true,
        text: lines.join("\n"),
        url: "https://timeapi.io",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const location = extractLocationPhrase(query) ?? query.trim();
    const place = await geocodePlace(location);
    if (!place?.timezone) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצא אזור זמן: ${location}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const time = await fetchTimezone(place.timezone);
    const d = time.currentLocalTime ? new Date(time.currentLocalTime) : null;
    const dayIdx = d?.getDay() ?? 0;
    const lines = [
      `מיקום: ${formatPlaceLabel(place)}`,
      `אזור זמן: ${time.timeZone ?? place.timezone}`,
      `שעה מקומית: ${formatLocalTime(time.currentLocalTime)}`,
      `UTC offset: ${formatOffset(time.currentUtcOffset?.seconds)}${time.isDayLightSavingActive ? " (DST פעיל)" : ""}`,
      ...(d ? [`יום: ${DAY_HE[dayIdx] ?? dayIdx}`] : []),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://timeapi.io/timezone/${encodeURIComponent(place.timezone)}`,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
