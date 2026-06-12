import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

export const fetchSpaceWeatherSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "noaa-space" as const;
  const label = "מזג אוויר חללי (NOAA)";
  try {
    const kp = await fetchJson<string[][]>(
      "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    );
    const latest = kp.at(-1);
    const kpVal = latest?.[1] ?? "—";
    const lines = [
      "NOAA Space Weather Prediction Center",
      `אינדקס Kp נוכחי: ${kpVal}`,
      kpVal !== "—" && Number(kpVal) >= 5 ? "⚠ סערה גיאומגנטית אפשרית" : "רמת פעילות: רגילה/מתונה",
      "מקור: services.swpc.noaa.gov",
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.filter(Boolean).join("\n"),
      url: "https://www.swpc.noaa.gov",
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
