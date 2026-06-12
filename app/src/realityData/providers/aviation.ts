import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type AdsbResponse = {
  ac?: Array<{ flight?: string; alt_baro?: number; gs?: number; track?: number; hex?: string }>;
};

const ISRAEL_RE = /ישראל|israel|tel\s*aviv|תל\s*אביב/i;

/** Live aircraft near Israel — airplanes.live (CORS-friendly). */
export const fetchAviationSearch = async (
  query: string,
  recentUserText: string[] = [],
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "adsb-aviation" as const;
  const label = "תעופה (ADS-B)";
  try {
    const context = [query, ...recentUserText].join(" ");
    const isIsrael = ISRAEL_RE.test(context);
    const lat = isIsrael ? 32.08 : 40.7;
    const lon = isIsrael ? 34.78 : -74.0;
    const data = await fetchJson<AdsbResponse>(
      `https://api.airplanes.live/v2/point/${lat}/${lon}/250`,
    );
    const ac = data.ac ?? [];
    if (!ac.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין מטוסים בטווח",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const lines = [
      `אזור: ${isIsrael ? "ישראל (מרכז)" : "ברירת מחדל"}`,
      `מטוסים בטווח 250km: ${ac.length}`,
      ...ac.slice(0, 12).map((a, i) => {
        const fl = (a.flight ?? "—").trim();
        const alt = a.alt_baro != null ? `${a.alt_baro}ft` : "—";
        const spd = a.gs != null ? `${Math.round(a.gs)}kn` : "—";
        return `${i + 1}. ${fl || "לא ידוע"} · גובה ${alt} · ${spd}`;
      }),
    ];
    if (/צבאי|military|מהם.*מטוס/i.test(query)) {
      lines.push(
        "הערה: ADS-B ציבורי לא מסמן באופן אמין מטוסים צבאיים; אין ספירה מדויקת של צבאי/אזרחי מהמקור הזה.",
      );
    }
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://api.airplanes.live",
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
