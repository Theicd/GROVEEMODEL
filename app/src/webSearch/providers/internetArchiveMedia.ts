import { buildMediaSearchQuery, buildMoviesSearchQuery } from "../intents";
import type { SearchSourceResult } from "../types";
import { searchArchiveMediaHits } from "./internetArchiveSearch";

const emptyResult = (error: string, started: number): SearchSourceResult => ({
  provider: "internet-archive-media",
  label: "Internet Archive · וידאו",
  ok: false,
  text: "",
  error,
  latencyMs: Math.round(performance.now() - started),
});

export const fetchInternetArchiveMediaSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "internet-archive-media" as const;
  const label = "Internet Archive · וידאו";

  const cleaned =
    buildMediaSearchQuery(query) || buildMoviesSearchQuery(query) || query.trim();
  if (!cleaned || cleaned.length < 2) {
    return emptyResult("הקלד מילות חיפוש לווידאו בארכיון (לפחות 2 תווים).", started);
  }

  try {
    const hits = await searchArchiveMediaHits(query, 16);
    if (!hits.length) {
      return emptyResult(`לא נמצאו סרטונים נגנים בארכיון עבור «${cleaned}».`, started);
    }

    const lines = [
      `שאילתה: ${cleaned} · Internet Archive`,
      ...hits.map((h, i) => `${i + 1}. ${h.title}${h.durationSec ? ` · ${h.durationSec}s` : ""}`),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: hits[0]?.url,
      mediaHits: hits,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return emptyResult(err instanceof Error ? err.message : "שגיאה בחיפוש Internet Archive", started);
  }
};
