import { describe, expect, it } from "vitest";
import type { SearchSourceResult } from "./types";

const CINEMA_SNIPPET =
  "קופה ראשית: הסרט · 91 ; צעצוע של סיפור 5-מדובב · 102 ; ספיידרמן: יום חדש · 150";

const scavioHit = (query: string, snippet: string): SearchSourceResult => ({
  provider: "scavio",
  label: "Scavio Google (web)",
  ok: true,
  text: `תוצאות Google (Scavio) · ${query}:`,
  webHits: [
    {
      id: "s1",
      title: "עכשיו בקולנוע",
      url: "https://www.cinema-city.co.il/movies",
      snippet,
    },
  ],
  latencyMs: 10,
});

describe("web provider merge", () => {
  it("keeps cinema-city listing when merging multiple scavio sub-queries", async () => {
    const { extractCinemaMoviesFromSources } = await import("./cinemaIlExtract");

    const weak = scavioHit("הסרטים המצליחים בקולנוע", "עכשיו בקולנוע בקרוב בקולנוע");
    const strong = scavioHit("סרטים בקולנוע ישראל עכשיו", CINEMA_SNIPPET);

    const mergedHits = [
      ...(weak.webHits ?? []),
      ...(strong.webHits ?? []),
    ];
    const merged: SearchSourceResult = {
      ...strong,
      webHits: mergedHits,
      text: `${weak.text}\n${strong.text}`,
    };

    const movies = extractCinemaMoviesFromSources([merged], 3);
    expect(movies.map((m) => m.title)).toEqual(
      expect.arrayContaining(["ספיידרמן: יום חדש", "צעצוע של סיפור 5 (מדובב)"]),
    );
  });
});
