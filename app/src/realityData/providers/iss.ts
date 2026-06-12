import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type IssPos = { latitude: number; longitude: number; altitude: number; velocity: number; timestamp: number };

export const fetchIssSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "iss-tracker" as const;
  const label = "תחנת חלל (ISS)";
  try {
    const data = await fetchJson<IssPos>("https://api.wheretheiss.at/v1/satellites/25544");
    const lines = [
      `מיקום ISS (זמן אמת):`,
      `קו רוחב: ${data.latitude.toFixed(2)}°`,
      `קו אורך: ${data.longitude.toFixed(2)}°`,
      `גובה: ${data.altitude.toFixed(0)} km`,
      `מהירות: ${data.velocity.toFixed(0)} km/h`,
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://api.wheretheiss.at",
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
