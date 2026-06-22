import { describe, expect, it } from "vitest";
import {
  extractCinemaMoviesFromSources,
  isCinemaHomepageHit,
  parseCinemaMoviesFromText,
} from "./cinemaIlExtract";
import { buildOpenWebTopicReply } from "./capabilityReplyMessages";
import type { SearchSourceResult } from "./types";

const CINEMA_SNIPPET =
  "קופה ראשית: הסרט-מתורגם לצרפתית · 91 ; קופה ראשית: הסרט · 91 ; צעצוע של סיפור 5-מדובב · 102 ; ספיידרמן: יום חדש · 150";

describe("cinemaIlExtract", () => {
  it("parses movie titles from cinema-city style snippets", () => {
    const titles = parseCinemaMoviesFromText(CINEMA_SNIPPET);
    expect(titles).toContain("צעצוע של סיפור 5 (מדובב)");
    expect(titles).toContain("ספיידרמן: יום חדש");
    expect(titles).not.toContain("הסרט");
  });

  it("flags homepage hits", () => {
    expect(
      isCinemaHomepageHit({
        title: "HOT CINEMA רשת בתי הקולנוע של ישראל | הוט סינמה",
        url: "https://hotcinema.co.il/",
        snippet: "עכשיו בקולנוע בקרוב בקולנוע",
      }),
    ).toBe(true);
    expect(
      isCinemaHomepageHit({
        title: "עכשיו בקולנוע",
        url: "https://www.cinema-city.co.il/movies",
        snippet: CINEMA_SNIPPET,
      }),
    ).toBe(false);
  });

  it("buildOpenWebTopicReply uses parsed listings not homepages", () => {
    const src: SearchSourceResult = {
      provider: "scavio",
      label: "Scavio Google (web)",
      ok: true,
      text: [
        "תוצאות Google (Scavio):",
        "1. עכשיו בקולנוע · https://www.cinema-city.co.il/movies",
        `   ${CINEMA_SNIPPET}`,
      ].join("\n"),
      webHits: [
        {
          title: "עכשיו בקולנוע",
          url: "https://www.cinema-city.co.il/movies",
          snippet: CINEMA_SNIPPET,
        },
        {
          title: "HOT CINEMA רשת בתי הקולנוע של ישראל",
          url: "https://hotcinema.co.il/",
          snippet: "עכשיו בקולנוע בקרוב בקולנוע",
        },
      ],
      latencyMs: 50,
    };
    const q =
      "חפש באינטרנט: מהם 3 הסרטים הכי מצליחים שמציגים עכשיו בבתי הקולנוע בישראל? תן תקציר";
    const movies = extractCinemaMoviesFromSources([src], 3);
    expect(movies.length).toBeGreaterThanOrEqual(2);
    const reply = buildOpenWebTopicReply(q, [src]);
    expect(reply).toMatch(/ספיידרמן|צעצוע של סיפור/);
    expect(reply).not.toMatch(/HOT CINEMA רשת|עמוד הבית|לא רק קולנוע/);
  });
});
