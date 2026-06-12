import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type GdacsFeature = { properties?: { eventname?: string; alertlevel?: string; country?: string } };

export const fetchDisasterSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "gdacs-disasters" as const;
  const label = "אסונות (GDACS)";
  try {
    const year = new Date().getFullYear();
    const data = await fetchJson<{ features?: GdacsFeature[] }>(
      `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=EQ;TC;FL;VO;WF&fromDate=${year - 1}-01-01&toDate=${year}-12-31&alertlevel=Green;Orange;Red`,
    );
    const feats = data.features ?? [];
    const active = feats.filter((f) => /orange|red/i.test(f.properties?.alertlevel ?? ""));
    const list = (active.length ? active : feats).slice(0, 8);
    if (!list.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין אירועים",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const lines = [
      "אירועי טבע (GDACS):",
      ...list.map((f, i) => {
        const p = f.properties ?? {};
        return `${i + 1}. ${p.eventname ?? "—"} · ${p.country ?? ""} · ${p.alertlevel ?? ""}`;
      }),
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://www.gdacs.org",
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
