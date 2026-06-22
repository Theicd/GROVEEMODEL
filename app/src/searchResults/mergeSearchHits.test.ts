import { describe, expect, it } from "vitest";

import type { GroveeNewsCard } from "../groveeNews/types";

import type { SearchSourceResult } from "../webSearch/types";

import { buildUnifiedSearchPayload, mergeSourcesToHits, newsCardToHit } from "./mergeSearchHits";

import { parseArxivText, parseGithubLines } from "./parseProviderLines";

import { cleanDisplaySnippet } from "./snippetCleanup";



describe("mergeSearchHits", () => {

  it("converts RSS cards to unified hits", () => {

    const card: GroveeNewsCard = {

      id: "c1",

      title: "כותרת בדיקה",

      titleOriginal: "Test",

      source: "ynet",

      sourceKey: "ynet",

      url: "https://www.ynet.co.il/article/1",

      image: "https://example.com/img.jpg",

      score: 20,

      publishedTs: 1_700_000_000_000,

      summary: "תקציר קצר",

    };

    const hit = newsCardToHit(card, 0);

    expect(hit.kind).toBe("rss");

    expect(hit.summarizable).toBe(true);

    expect(hit.faviconUrl).toContain("favicons");

    expect(hit.meta?.engine).toBe("RSS");

  });



  it("parses GitHub repo lines without duplicating title in snippet", () => {

    const sources: SearchSourceResult[] = [

      {

        provider: "github",

        label: "GitHub",

        ok: true,

        text: "שאילתה: robotics\n1. openai/robotics [Python]: Humanoid control stack (https://github.com/openai/robotics) ★1,234",

        latencyMs: 100,

      },

    ];

    const hits = mergeSourcesToHits(sources);

    expect(hits).toHaveLength(1);

    expect(hits[0].kind).toBe("github");

    expect(hits[0].title).toBe("openai/robotics: Humanoid control stack");

    expect(hits[0].snippet).toBe("");

    expect(hits[0].url).toContain("github.com");

  });



  it("parses arXiv multi-line blocks", () => {

    const text = [

      "חיפוש arXiv: robotics",

      "1. Deep RL for robots (2024-03-01)",

      "   https://arxiv.org/abs/2403.00001",

      "   We study locomotion policies for legged robots…",

      "2. Vision transformers (2024-02-15)",

      "   https://arxiv.org/abs/2402.00002",

    ].join("\n");

    const hits = parseArxivText(text);

    expect(hits).toHaveLength(2);

    expect(hits[0].title).toBe("Deep RL for robots");

    expect(hits[0].snippet).toContain("locomotion");

    expect(hits[0].url).toContain("arxiv.org");

  });



  it("merges web hits from SearXNG provider", () => {

    const sources: SearchSourceResult[] = [

      {

        provider: "searxng",

        label: "SearXNG",

        ok: true,

        text: "",

        webHits: [

          {

            id: "w1",

            title: "Robotics trends",

            url: "https://example.com/robotics",

            snippet: "Latest in robotics",

            engine: "google",

          },

        ],

        latencyMs: 200,

      },

    ];

    const payload = buildUnifiedSearchPayload("robotics", sources);

    expect(payload.facets.web).toBe(1);

    expect(payload.hits[0].kind).toBe("web");

    expect(payload.hits[0].summarizable).toBe(true);

  });



  it("promotes YouTube web hits to youtube kind", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "searxng",
        label: "SearXNG",
        ok: true,
        text: "",
        webHits: [
          {
            id: "w-yt",
            title: "Demo song",
            url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            snippet: "Official video",
            engine: "google",
          },
        ],
        latencyMs: 100,
      },
    ];
    const payload = buildUnifiedSearchPayload("שיר demo", sources);
    expect(payload.facets.youtube).toBe(1);
    expect(payload.facets.web).toBe(0);
    expect(payload.hits[0].kind).toBe("youtube");
    expect(payload.hits[0].mediaPlayUrl).toContain("embed");
  });



  it("sets preferVideoFilter for artist name queries with YouTube hits", () => {
    const payload = buildUnifiedSearchPayload("שלמה ארצי", [
      {
        provider: "invidious-videos",
        label: "YouTube",
        ok: true,
        text: "",
        latencyMs: 80,
        mediaHits: [
          {
            id: "inv-1",
            mediaType: "video",
            youtubeSubType: "video",
            title: "שלמה ארצי - לי לבד",
            url: "https://www.youtube.com/watch?v=abc12345678",
            playUrl: "https://www.youtube-nocookie.com/embed/abc12345678",
            thumbnail: "https://i.ytimg.com/vi/abc12345678/hqdefault.jpg",
            source: "YouTube",
          },
        ],
      },
    ]);
    expect(payload.preferVideoFilter).toBe(true);
    expect(payload.facets.youtube).toBe(1);
  });

  it("sets preferVideoFilter for music queries with YouTube hits", () => {
    const payload = buildUnifiedSearchPayload("שיר עומר אדם", [
      {
        provider: "invidious-videos",
        label: "YouTube",
        ok: true,
        text: "",
        latencyMs: 80,
        mediaHits: [
          {
            id: "inv-1",
            mediaType: "video",
            youtubeSubType: "video",
            title: "עומר אדם - שיר",
            url: "https://www.youtube.com/watch?v=abc12345678",
            playUrl: "https://www.youtube-nocookie.com/embed/abc12345678",
            thumbnail: "https://i.ytimg.com/vi/abc12345678/hqdefault.jpg",
            source: "YouTube",
          },
        ],
      },
    ]);
    expect(payload.preferVideoFilter).toBe(true);
    expect(payload.facets.youtube).toBe(1);
  });

  it("sets preferRssFilter for news queries with RSS hits", () => {
    const sources: SearchSourceResult[] = [

      {

        provider: "grovee-news",

        label: "news",

        ok: true,

        text: "x",

        newsCards: [

          {

            id: "1",

            title: "A",

            titleOriginal: "A",

            source: "ynet",

            sourceKey: "ynet",

            url: "https://www.ynet.co.il/a",

            image: "",

            score: 10,

            publishedTs: 0,

          },

        ],

        latencyMs: 1,

      },

    ];

    const payload = buildUnifiedSearchPayload("חפש חדשות על כלכלה", sources);

    expect(payload.preferRssFilter).toBe(true);

    expect(payload.facets.rss).toBe(1);

  });



  it("dedupes by URL keeping the higher-scored hit", () => {

    const sources: SearchSourceResult[] = [

      {

        provider: "grovee-news",

        label: "news",

        ok: true,

        text: "x",

        newsCards: [

          {

            id: "1",

            title: "A",

            titleOriginal: "A",

            source: "ynet",

            sourceKey: "ynet",

            url: "https://example.com/same",

            image: "",

            score: 10,

            publishedTs: 0,

          },

        ],

        latencyMs: 1,

      },

      {

        provider: "searxng",

        label: "web",

        ok: true,

        text: "",

        webHits: [

          {

            id: "w1",

            title: "A web",

            url: "https://example.com/same",

            snippet: "dup",

          },

        ],

        latencyMs: 1,

      },

    ];

    const payload = buildUnifiedSearchPayload("q", sources);

    expect(payload.hits).toHaveLength(1);

    expect(payload.hits[0].kind).toBe("web");

  });

});



describe("snippetCleanup", () => {

  it("drops snippets that repeat the title", () => {
    expect(cleanDisplaySnippet("Same title", "Same title")).toBe("");
  });



  it("parseGithubLines keeps description only in title", () => {
    const hits = parseGithubLines(
      "1. foo/bar [Rust]: A cool crate (https://github.com/foo/bar) ★99",
    );
    expect(hits[0].title).toBe("foo/bar: A cool crate");
    expect(hits[0].snippet).toBe("");
  });

  it("parses Wikipedia text into web hits", () => {
    const text = [
      "- Police (מלא):",
      "A police force is constituted body of persons empowered by the state.",
      "  IMAGE: https://upload.wikimedia.org/wikipedia/commons/thumb/police.jpg",
      "  https://en.wikipedia.org/wiki/Police",
      "",
      "- Police officer: A warranted employee of a police force.",
      "  https://en.wikipedia.org/wiki/Police_officer",
    ].join("\n");
    const hits = mergeSourcesToHits([
      {
        provider: "wikipedia-en",
        label: "Wikipedia",
        ok: true,
        text,
        latencyMs: 50,
      },
    ]);
    expect(hits.length).toBeGreaterThanOrEqual(1);
    expect(hits[0].kind).toBe("web");
    expect(hits[0].url).toContain("wikipedia.org");
    expect(hits[0].imageUrl).toContain("wikimedia.org");
  });

  it("converts Internet Archive movie with playUrl to in-app video hit", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "movie-catalog",
        label: "סרטים וסדרות",
        ok: true,
        text: "",
        latencyMs: 100,
        movieHits: [
          {
            id: "archive-TikTok-7449350901090798855",
            title: "תיעוד מרדף",
            year: 2024,
            url: "https://archive.org/details/TikTok-7449350901090798855",
            snippet: "מרדף בשומרון",
            poster: "https://archive.org/services/img/TikTok-7449350901090798855",
            source: "Internet Archive",
            playUrl: "https://archive.org/download/TikTok-7449350901090798855/clip.mp4",
            durationSec: 95,
          },
        ],
      },
    ];
    const hits = mergeSourcesToHits(sources);
    expect(hits).toHaveLength(1);
    expect(hits[0].kind).toBe("video");
    expect(hits[0].mediaPlayUrl).toContain("archive.org/download");
    expect(hits[0].sourceLabel).toBe("Internet Archive");
    expect(hits[0].durationSec).toBe(95);
  });

  it("converts movie catalog hits to unified hits", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "movie-catalog",
        label: "סרטים וסדרות",
        ok: true,
        text: "שאילתה: Inception",
        latencyMs: 200,
        movieHits: [
          {
            id: "yts-1",
            title: "Inception",
            year: 2010,
            url: "https://www.youtube.com/watch?v=abc12345678",
            snippet: "A thief who steals secrets through dreams.",
            poster: "https://image.tmdb.org/t/p/w342/poster.jpg",
            ageRating: "PG-13",
            runtime: 148,
            quality: "1080p",
            seeds: 120,
            source: "YTS",
          },
        ],
      },
    ];
    const hits = mergeSourcesToHits(sources);
    expect(hits).toHaveLength(1);
    expect(hits[0].kind).toBe("youtube");
    expect(hits[0].title).toContain("Inception");
    expect(hits[0].imageUrl).toContain("tmdb.org");
    expect(hits[0].summarizable).toBe(false);
  });

  it("sets preferMoviesFilter for movie queries", () => {
    const payload = buildUnifiedSearchPayload("סרט Inception", [
      {
        provider: "movie-catalog",
        label: "סרטים וסדרות",
        ok: true,
        text: "",
        latencyMs: 100,
        movieHits: [
          {
            id: "m1",
            title: "Inception",
            url: "https://example.com",
            snippet: "test",
          },
        ],
      },
    ]);
    expect(payload.preferMoviesFilter).toBe(true);
    expect(payload.facets.movies).toBe(1);
  });

  it("converts Pixabay media hits to image and video hits", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "pixabay-images",
        label: "Pixabay",
        ok: true,
        text: "",
        latencyMs: 100,
        mediaHits: [
          {
            id: "pixabay-img-1",
            mediaType: "image",
            title: "cat",
            url: "https://pixabay.com/photos/cat-1/",
            playUrl: "https://cdn.pixabay.com/photo/large.jpg",
            thumbnail: "https://cdn.pixabay.com/photo/small.jpg",
            author: "user1",
            licenseUrl: "https://pixabay.com/photos/cat-1/",
            tags: "cat, pet",
            source: "Pixabay",
          },
        ],
      },
      {
        provider: "pixabay-videos",
        label: "Pixabay",
        ok: true,
        text: "",
        latencyMs: 120,
        mediaHits: [
          {
            id: "pixabay-vid-2",
            mediaType: "video",
            title: "ocean",
            url: "https://pixabay.com/videos/ocean-2/",
            playUrl: "https://cdn.pixabay.com/video/medium.mp4",
            thumbnail: "https://cdn.pixabay.com/video/thumb.jpg",
            durationSec: 42,
            source: "Pixabay",
          },
        ],
      },
    ];
    const hits = mergeSourcesToHits(sources);
    expect(hits.filter((h) => h.kind === "image")).toHaveLength(1);
    expect(hits.filter((h) => h.kind === "video")).toHaveLength(1);
    expect(hits.find((h) => h.kind === "video")?.durationSec).toBe(42);
  });

  it("sets preferProductsFilter for product queries", () => {
    const payload = buildUnifiedSearchPayload("חלב תנובה", [
      {
        provider: "israeli-products",
        label: "מוצרים",
        ok: true,
        text: "",
        latencyMs: 50,
        productHits: [
          {
            id: "p1",
            barcode: "7290004131074",
            title: "חלב 3% — תנובה",
            brand: "תנובה",
            url: "https://cheapersal.co.il/product/7290004131074",
            snippet: "₪6.90 · מקרר · ברקוד 7290004131074",
            imageUrl: "https://price-api.additlist.com/images/catalog/carrefour/7290004131074.jpg",
            source: "קטלוג ישראלי",
            priceNis: 6.9,
            priceSummary: "₪6.90 · הכי זול: רמי לוי",
          },
        ],
      },
    ]);
    expect(payload.preferProductsFilter).toBe(true);
    expect(payload.facets.products).toBe(1);
    const row = payload.hits.find((h) => h.kind === "product");
    expect(row?.imageUrl).toMatch(/additlist/i);
    expect(row?.url).toMatch(/cheapersal\.co\.il/i);
    expect(row?.url).not.toMatch(/openfoodfacts/i);
    expect(row?.meta?.priceNis).toBe(6.9);
    expect(row?.snippet).toContain("₪6.90");
  });

  it("sets preferImagesFilter for image queries", () => {
    const payload = buildUnifiedSearchPayload("תמונות חתול", [
      {
        provider: "pixabay-images",
        label: "Pixabay",
        ok: true,
        text: "",
        latencyMs: 80,
        mediaHits: [
          {
            id: "i1",
            mediaType: "image",
            title: "cat",
            url: "https://example.com",
            playUrl: "https://example.com/l.jpg",
            thumbnail: "https://example.com/s.jpg",
          },
        ],
      },
    ]);
    expect(payload.preferImagesFilter).toBe(true);
    expect(payload.facets.images).toBe(1);
  });

  it("ranks Columbus Wikipedia above irrelevant RSS in unified payload", () => {
    const payload = buildUnifiedSearchPayload("columbus", [
      {
        provider: "grovee-news",
        label: "חדשות",
        ok: true,
        text: "",
        latencyMs: 40,
        newsCards: [
          {
            id: "gh",
            title: "Local sports update",
            titleOriginal: "Local sports update",
            source: "GhanaWeb",
            sourceKey: "gh_ghanaweb",
            url: "https://www.ghanaweb.com/GhanaHomePage/sports/1",
            image: "",
            score: 80,
            publishedTs: Date.now() / 1000,
          },
        ],
      },
      {
        provider: "wikipedia-en",
        label: "Wikipedia",
        ok: true,
        latencyMs: 30,
        text: "- Christopher Columbus (מלא):\nItalian explorer.\nhttps://en.wikipedia.org/wiki/Christopher_Columbus",
      },
    ]);
    expect(payload.hits.some((h) => /ghanaweb/i.test(h.url))).toBe(false);
    expect(payload.hits[0]?.title).toMatch(/Columbus/i);
  });
});

