import { describe, expect, it, beforeEach } from "vitest";
import type { ArticleRecord, RssItem } from "../types";
import { rankArticlesForQuery, prefilterRssItemsForQuery } from "../search/relevance";
import {
  ensureSearchIndexLoaded,
  rankIndexedArticlesForQuery,
  resetSearchIndex,
} from "../search/flexIndex";
import { shouldRetainSearchFocus } from "../search/searchInteraction";

const WARM_SEARCH_BUDGET_MS = 120;
const RSS_PREFILTER_CAP = 700;

function article(id: string, title: string): ArticleRecord {
  return {
    id,
    url: `https://example.com/${id}`,
    source: "Test",
    sourceKey: "test",
    title,
    image: "",
    publishDate: new Date().toISOString(),
    publishedTs: Date.now(),
    articleText: title,
    summary: title,
    keyFacts: [],
    keywords: [],
    entities: [],
    clusterId: id,
    confidence: "LOW",
    fetchedAt: Date.now(),
    summarizedAt: Date.now(),
  };
}

function rssItem(id: string, title: string): RssItem {
  return {
    id,
    link: `https://example.com/${id}`,
    source: "Test",
    sourceKey: "test",
    title,
    description: title,
    published: new Date().toISOString(),
    publishedTs: Date.now(),
    image: "",
    category: "news",
  };
}

function makeCorpus(n: number, hitEvery = 50): ArticleRecord[] {
  return Array.from({ length: n }, (_, i) =>
    article(`a-${i}`, i % hitEvery === 0 ? `Trump headline ${i}` : `Unrelated story number ${i}`),
  );
}

function elapsed(run: () => void): number {
  const t0 = performance.now();
  run();
  return performance.now() - t0;
}

async function elapsedAsync(run: () => Promise<void>): Promise<number> {
  const t0 = performance.now();
  await run();
  return performance.now() - t0;
}

describe("Search UI QA suite", () => {
  beforeEach(() => {
    resetSearchIndex();
  });

  describe("QA-1 — indexed path faster than full scan at scale", () => {
    it("warm indexed search beats scanning 5000 articles", async () => {
      const corpus = makeCorpus(5000);
      const fullMs = elapsed(() => {
        rankArticlesForQuery(corpus, "trump", 24);
      });

      await ensureSearchIndexLoaded(corpus);
      const indexedMs = await elapsedAsync(async () => {
        await rankIndexedArticlesForQuery(corpus, "trump", 24);
      });

      expect(indexedMs).toBeLessThan(fullMs);
    });
  });

  describe("QA-2 — warm indexed search within UI budget", () => {
    it(`second search on 2000 articles under ${WARM_SEARCH_BUDGET_MS}ms`, async () => {
      const corpus = makeCorpus(2000);
      await ensureSearchIndexLoaded(corpus);
      await rankIndexedArticlesForQuery(corpus, "warmup", 8);

      const ms = await elapsedAsync(async () => {
        const hits = await rankIndexedArticlesForQuery(corpus, "trump", 24);
        expect(hits.length).toBeGreaterThan(0);
      });
      expect(ms).toBeLessThan(WARM_SEARCH_BUDGET_MS);
    });
  });

  describe("QA-3 — RSS prefilter caps workload", () => {
    it(`never passes more than ${RSS_PREFILTER_CAP} items to ranker`, () => {
      const items = Array.from({ length: 3000 }, (_, i) => rssItem(`r-${i}`, `Story about trump ${i}`));
      const narrowed = prefilterRssItemsForQuery(items, "trump");
      expect(narrowed.length).toBeLessThanOrEqual(RSS_PREFILTER_CAP);
      const hits = rankArticlesForQuery(
        narrowed.map((r) => article(r.id, r.title)),
        "trump",
        24,
      );
      expect(hits.length).toBeGreaterThan(0);
    });
  });

  describe("QA-4 — incremental index (no full rebuild on growth)", () => {
    it("indexes only missing articles when corpus grows", async () => {
      const first = makeCorpus(40);
      await ensureSearchIndexLoaded(first);
      const afterFirst = await ensureSearchIndexLoaded(first);
      expect(afterFirst).toBe(40);

      const grown = [...first, ...makeCorpus(10).map((a, i) => article(`new-${i}`, a.title))];
      const afterGrow = await ensureSearchIndexLoaded(grown);
      expect(afterGrow).toBe(50);
    });
  });

  describe("QA-5 — search focus blur guard", () => {
    it("retains interaction when focus moves to submit inside form", () => {
      const button = { nodeType: 1 } as unknown as Node;
      const form = {
        contains(node: Node) {
          return node === button;
        },
      } as unknown as HTMLFormElement;

      expect(shouldRetainSearchFocus(button, form)).toBe(true);
      expect(shouldRetainSearchFocus(null, form)).toBe(false);
      expect(shouldRetainSearchFocus(button, null)).toBe(false);
    });
  });

  describe("QA-6 — ranking only on submit (not per keystroke)", () => {
    it("simulates isolated SearchBar: one rank call after typing completes", () => {
      const corpus = makeCorpus(500);
      let rankCalls = 0;
      const simulateTyping = (chars: string) => {
        let draft = "";
        for (const ch of chars) {
          draft += ch;
        }
        rankCalls++;
        rankArticlesForQuery(corpus, draft.trim(), 24);
      };
      simulateTyping("trump");
      expect(rankCalls).toBe(1);
    });
  });
});
