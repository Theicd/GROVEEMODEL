// @ts-nocheck
import { Document } from "flexsearch";
import { rankArticlesForQuery, type RankOptions } from "./relevance";
import type { ArticleRecord } from "../types";

type RawSearchHit = { id: string; score: number };
import { yieldToMain } from "../util/yieldToMain";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let index: Document<any, any> | null = null;
let indexedIds = new Set<string>();

export function resetSearchIndex(): void {
  index = null;
  indexedIds = new Set();
}

function ensureIndex(): Document<any, any> {
  if (!index) {
    index = new Document({
      document: {
        id: "id",
        index: ["title", "summary", "keywords", "entities", "articleText"],
      },
      tokenize: "forward",
      cache: true,
    });
  }
  return index;
}

const INDEX_CHUNK = 40;

export async function indexArticles(articles: ArticleRecord[]): Promise<void> {
  const doc = ensureIndex();
  let added = 0;
  for (const a of articles) {
    if (indexedIds.has(a.id)) continue;
    doc.add({
      id: a.id,
      title: a.title,
      summary: a.summary,
      keywords: (a.keywords ?? []).join(" "),
      entities: (a.entities ?? []).join(" "),
      articleText: a.articleText.slice(0, 4000),
    });
    indexedIds.add(a.id);
    added++;
    if (added % INDEX_CHUNK === 0) await yieldToMain();
  }
}

/** Incrementally index new articles only — avoids full rebuild on every poll. */
export async function ensureSearchIndexLoaded(articles: ArticleRecord[]): Promise<number> {
  if (!articles.length) return 0;
  const missing = articles.filter((a) => !indexedIds.has(a.id));
  if (missing.length) await indexArticles(missing);
  return indexedIds.size;
}

export function getSearchIndexSize(): number {
  return indexedIds.size;
}

function normalizeSearchResults(res: unknown): Array<{ field?: string; result: unknown[] }> {
  if (Array.isArray(res)) return res as Array<{ field?: string; result: unknown[] }>;
  if (res && typeof res === "object" && "result" in res) {
    const inner = (res as { result: unknown }).result;
    if (Array.isArray(inner)) return inner as Array<{ field?: string; result: unknown[] }>;
  }
  return [];
}

export async function searchIndex(query: string, limit = 24): Promise<RawSearchHit[]> {
  const doc = ensureIndex();
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 1);

  const expanded = terms.length ? terms : [query.toLowerCase().trim()].filter(Boolean);
  const scoreMap = new Map<string, number>();

  for (const term of expanded) {
    const raw = await doc.searchAsync(term, { limit: limit * 3 });
    const fields = normalizeSearchResults(raw);
    for (const field of fields) {
      const ids = field.result ?? [];
      for (const id of ids) {
        const sid = String(id);
        scoreMap.set(sid, (scoreMap.get(sid) ?? 0) + 1);
      }
    }
  }

  return [...scoreMap.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

/** FlexSearch candidates + relevance scoring on a small subset (fast for large corpora). */
export async function rankIndexedArticlesForQuery(
  articles: ArticleRecord[],
  query: string,
  limit = 24,
  options: RankOptions = {},
) {
  const q = query.trim();
  if (!q || !articles.length) return [];

  await ensureSearchIndexLoaded(articles);
  const raw = await searchIndex(q, Math.max(limit * 4, 64));
  const byId = new Map(articles.map((a) => [a.id, a]));
  const seen = new Set<string>();
  const candidates: ArticleRecord[] = [];

  for (const hit of raw) {
    const article = byId.get(hit.id);
    if (!article || seen.has(article.id)) continue;
    seen.add(article.id);
    candidates.push(article);
  }

  if (candidates.length < limit) {
    for (const article of articles) {
      if (seen.has(article.id)) continue;
      candidates.push(article);
      if (candidates.length >= Math.min(articles.length, 400)) break;
    }
  }

  return rankArticlesForQuery(candidates, q, limit, options);
}

/** @deprecated use rankArticlesForQuery from relevance.ts */
export function fallbackArticleSearch(articles: ArticleRecord[], query: string): RawSearchHit[] {
  return rankArticlesForQuery(articles, query).map((h) => ({ id: h.id, score: h.score }));
}

export function expandQueryKeywords(query: string): string[] {
  return query
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1);
}
