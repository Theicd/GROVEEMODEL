import { fetchText } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";
import {
  countTleSatellites,
  formatStarlinkCatalogText,
  getFreshStarlinkCatalogCache,
  getSeedStarlinkCatalogCache,
  loadStarlinkCatalogCache,
  sampleNamesFromTle,
  saveStarlinkCatalogCache,
  STARLINK_CATALOG_MAX_AGE_MS,
} from "../../liveWorld/starlinkSnapshot";

type CelesTrakEntry = { OBJECT_NAME?: string; NORAD_CAT_ID?: number };

const STARLINK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle";
const ACTIVE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json";

const RELAY = (url: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;

const fetchStarlinkTleText = async (timeoutMs: number): Promise<string> => {
  try {
    return await fetchText(
      STARLINK_TLE_URL,
      { headers: { Accept: "text/plain, */*" } },
      { timeoutMs },
    );
  } catch {
    return fetchText(RELAY(STARLINK_TLE_URL), { headers: { Accept: "text/plain, */*" } }, { timeoutMs });
  }
};

const okResultFromCache = (
  cache: ReturnType<typeof getFreshStarlinkCatalogCache>,
  started: number,
  stale = true,
): SearchSourceResult => ({
  provider: "starlink-catalog",
  label: "Starlink (CelesTrak / עולם חי)",
  ok: true,
  text: formatStarlinkCatalogText(cache!, stale),
  url: STARLINK_TLE_URL,
  latencyMs: Math.round(performance.now() - started),
});

/** Starlink catalog count via CelesTrak GROUP=starlink TLE (lighter than full JSON). */
export const fetchStarlinkCatalogSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "starlink-catalog" as const;
  const label = "Starlink (CelesTrak / עולם חי)";

  const fresh = getFreshStarlinkCatalogCache();
  if (fresh) {
    return okResultFromCache(fresh, started, true);
  }

  const staleAny = loadStarlinkCatalogCache(STARLINK_CATALOG_MAX_AGE_MS * 48);
  if (staleAny) {
    return okResultFromCache(staleAny, started, true);
  }

  try {
    const tle = await fetchStarlinkTleText(8_000);
    const total = countTleSatellites(tle);
    if (total) {
      const cache = {
        fetchedAt: Date.now(),
        total,
        sample: sampleNamesFromTle(tle, 6),
      };
      saveStarlinkCatalogCache(cache);
      return {
        provider,
        label,
        ok: true,
        text: formatStarlinkCatalogText(cache, false),
        url: STARLINK_TLE_URL,
        latencyMs: Math.round(performance.now() - started),
      };
    }
  } catch {
    /* seed below */
  }

  void fetchStarlinkTleText(25_000)
    .then((tle) => {
      const total = countTleSatellites(tle);
      if (!total) return;
      saveStarlinkCatalogCache({
        fetchedAt: Date.now(),
        total,
        sample: sampleNamesFromTle(tle, 6),
      });
    })
    .catch(() => null);

  const seed = getSeedStarlinkCatalogCache();
  saveStarlinkCatalogCache(seed);
  return {
    provider,
    label,
    ok: true,
    text: [
      formatStarlinkCatalogText(seed, true),
      "הערה: CelesTrak איטי — ספירה משוערת ממאגר seed; רענון ברקע.",
    ].join("\n"),
    url: STARLINK_TLE_URL,
    latencyMs: Math.round(performance.now() - started),
  };
};

const parseCatalog = (data: unknown): CelesTrakEntry[] => (Array.isArray(data) ? data : []);

const sampleLines = (items: CelesTrakEntry[], n = 6): string[] =>
  items
    .slice(0, n)
    .map((s, i) => `${i + 1}. ${s.OBJECT_NAME ?? "—"} (NORAD ${s.NORAD_CAT_ID ?? "?"})`);

/** Active satellite catalog count via CelesTrak (same source as Live World layer). */
export const fetchSatelliteCatalogSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "celestrak" as const;
  const label = "לוויינים (CelesTrak)";

  try {
    const { fetchJson } = await import("../../webSearch/fetchJson");
    const data = await fetchJson<CelesTrakEntry[]>(ACTIVE_URL, undefined, { timeoutMs: 18_000 });
    const items = parseCatalog(data);
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

    const sample = sampleLines(items);
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

export { clearStarlinkCatalogCacheForTests } from "../../liveWorld/starlinkSnapshot";
