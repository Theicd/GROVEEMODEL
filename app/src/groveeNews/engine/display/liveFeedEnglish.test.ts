import { describe, expect, it, vi } from "vitest";
import type { ArticleRecord } from "../types";

vi.mock("../translate/googleTranslate", () => ({
  translateTexts: vi.fn(async (texts: string[], target: string) => ({
    texts: texts.map((t) => `[${target}] ${t}`),
    provider: "cache" as const,
  })),
}));

vi.mock("../settings/userNewsProfile", () => ({
  getUserNewsProfile: () => ({ uiLanguage: "en", locale: "en-US", pollTier: "core" }),
}));

import { applyDisplayLanguageBatch } from "./liveFeedDisplay";

describe("liveFeedDisplay", () => {
  it("translates non-English RSS headlines for display", async () => {
    const article: ArticleRecord = {
      id: "fr::1",
      url: "https://www.lemonde.fr/a",
      source: "Le Monde",
      sourceKey: "fr_lemonde",
      title: "La France annonce une réforme",
      image: "",
      publishDate: "",
      publishedTs: Date.now(),
      articleText: "Le gouvernement a présenté un plan.",
      summary: "Le gouvernement a présenté un plan.",
      keyFacts: [],
      keywords: [],
      entities: [],
      clusterId: "fr::1",
      confidence: "LOW",
      fetchedAt: Date.now(),
      summarizedAt: 0,
    };

    const [out] = await applyDisplayLanguageBatch([article], "en");
    expect(out.displayTitle).toBe("[en] La France annonce une réforme");
    expect(out.displaySummary).toContain("[en]");
  });

  it("translates English headlines to French when target is fr", async () => {
    const article: ArticleRecord = {
      id: "en::1",
      url: "https://news.site/a",
      source: "Sky News",
      sourceKey: "skynews",
      title: "Police launch probe after city centre incident",
      image: "",
      publishDate: "",
      publishedTs: Date.now(),
      articleText: "Officers said the investigation is ongoing.",
      summary: "Officers said the investigation is ongoing.",
      keyFacts: [],
      keywords: [],
      entities: [],
      clusterId: "en::1",
      confidence: "LOW",
      fetchedAt: Date.now(),
      summarizedAt: 0,
    };

    const [out] = await applyDisplayLanguageBatch([article], "fr");
    expect(out.displayTitle).toBe("[fr] Police launch probe after city centre incident");
  });

  it("skips English headlines when target is en", async () => {
    const article: ArticleRecord = {
      id: "en::2",
      url: "https://news.site/b",
      source: "Sky News",
      sourceKey: "skynews",
      title: "Police launch probe after city centre incident",
      image: "",
      publishDate: "",
      publishedTs: Date.now(),
      articleText: "Officers said the investigation is ongoing.",
      summary: "Officers said the investigation is ongoing.",
      keyFacts: [],
      keywords: [],
      entities: [],
      clusterId: "en::2",
      confidence: "LOW",
      fetchedAt: Date.now(),
      summarizedAt: 0,
    };

    const [out] = await applyDisplayLanguageBatch([article], "en");
    expect(out.displayTitle).toBe(article.title);
    expect(out.displaySummary).toBe(article.summary);
  });
});
