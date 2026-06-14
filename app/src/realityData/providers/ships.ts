import type { SearchSourceResult } from "../../webSearch/types";
import { aggregateShipHits, formatShipsText } from "../shipAggregate";
import { resolveShipRegion } from "../shipRegion";

/** Live AIS ships — Digitraffic + Globe cache + route markers (demo, labeled separately). */
export const fetchShipsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "ais-ships" as const;
  const label = "ספינות (AIS / עולם חי)";
  const region = await resolveShipRegion(query);

  try {
    const agg = await aggregateShipHits(query, region);

    if (!agg.allHits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error:
          region.bbox && /סואץ|suez/i.test(region.label)
            ? "אין דיווחי AIS חיים בתעלת סואץ — Digitraffic לא מכסה את האזור; פתח «עולם חי» לשכבת ספינות"
            : "לא נמצאו ספינות בטווח",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    return {
      provider,
      label,
      ok: true,
      text: formatShipsText(region.label, agg, query),
      url: "https://meri.digitraffic.fi/en/web/ais/vessels",
      latencyMs: Math.round(performance.now() - started),
    };
  } catch {
    try {
      const agg = await aggregateShipHits(query, region);
      if (agg.allHits.length) {
        return {
          provider,
          label,
          ok: true,
          text: formatShipsText(region.label, agg, query),
          url: "https://meri.digitraffic.fi/en/web/ais/vessels",
          latencyMs: Math.round(performance.now() - started),
        };
      }
    } catch {
      /* fall through */
    }

    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "Digitraffic AIS unavailable — try opening Reality Live ships layer",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
