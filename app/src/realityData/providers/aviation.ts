import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type AdsbResponse = {
  ac?: Array<{ flight?: string; alt_baro?: number; gs?: number; track?: number; hex?: string }>;
};

type OpenSkyResponse = { states?: Array<(string | number | null)[]> | null };

const ISRAEL_RE = /ישראל|israel|tel\s*aviv|תל\s*אביב/i;
const GLOBAL_RE = /בעולם|worldwide|global|ברחבי\s+העולם|around\s+the\s+world|in\s+the\s+air/i;

/** Live aircraft — airplanes.live (regional) or OpenSky (global count). */
export const fetchAviationSearch = async (
  query: string,
  recentUserText: string[] = [],
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "adsb-aviation" as const;
  const label = "תעופה (ADS-B)";
  const context = [query, ...recentUserText].join(" ");
  const wantGlobal = GLOBAL_RE.test(context);

  try {
    if (wantGlobal) {
      const data = await fetchJson<OpenSkyResponse>("https://opensky-network.org/api/states/all", undefined, {
        timeoutMs: 18_000,
      });
      const states = (data.states ?? []).filter((s) => s?.[5] != null && s?.[6] != null);
      if (!states.length) {
        return {
          provider,
          label,
          ok: false,
          text: "",
          error: "OpenSky לא החזיר מטוסים",
          latencyMs: Math.round(performance.now() - started),
        };
      }
      const lines = [
        "אזור: גלובלי (OpenSky ADS-B)",
        `מטוסים באוויר (דיווח אחרון): ${states.length}`,
        "הערה: עולם חי (🌐) מציג מטוסים על המפה — «הצג על המפה» לצפייה.",
        ...states.slice(0, 10).map((s, i) => {
          const call = String(s[1] ?? "—").trim() || "—";
          const alt = s[7] != null ? `${s[7]}m` : "—";
          const spd = s[9] != null ? `${Math.round(Number(s[9]))}m/s` : "—";
          return `${i + 1}. ${call} · גובה ${alt} · ${spd}`;
        }),
      ];
      return {
        provider,
        label,
        ok: true,
        text: lines.join("\n"),
        url: "https://opensky-network.org",
        latencyMs: Math.round(performance.now() - started),
      };
    }

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
      `אזור: ${isIsrael ? "ישראל (מרכז)" : "ברירת מחדל (NYC)"} · רדיוס 250km`,
      `מטוסים בטווח: ${ac.length}`,
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
