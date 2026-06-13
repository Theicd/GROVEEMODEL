import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type CelesTrakEntry = { OBJECT_NAME?: string; NORAD_CAT_ID?: number };

/** Active satellite catalog count via CelesTrak (same source as Live World layer). */
export const fetchSatelliteCatalogSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "celestrak" as const;
  const label = "לוויינים (CelesTrak)";

  try {
    const data = await fetchJson<CelesTrakEntry[]>(
      "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json",
      undefined,
      { timeoutMs: 18_000 },
    );
    const items = Array.isArray(data) ? data : [];
    const total = items.length;
    if (!total) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "קטalog ריק",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const sample = items.slice(0, 6).map((s, i) => `${i + 1}. ${s.OBJECT_NAME ?? "—"} (NORAD ${s.NORAD_CAT_ID ?? "?"})`);
    const lines = [
      `לוויינים פעילים בקטalog CelesTrak (GROUP=active): ${total}`,
      "עולם חי (🌐) עוקב אחר ~200 לוויינים עם מסלולים + ISS בזמן אמת — שכבת «לוויינים».",
      /iss|תחנת\s+החלל/i.test(query)
        ? `ISS נמדד בנפרד (WhereTheISS) — לא כלל ב-${total}.`
        : "לספירת ISS בנפרד — שאל על תחנת החלל הבינלאומית.",
      "דוגמאות:",
      ...sample,
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://celestrak.org/NORAD/elements/",
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
