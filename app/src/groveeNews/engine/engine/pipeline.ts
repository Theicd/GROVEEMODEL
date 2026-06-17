// @ts-nocheck
import { ensureEnglishDisplay } from "../display/englishDisplay";
import { applyDisplayLanguageBatch } from "../display/liveFeedDisplay";
import { ingestExternalSources } from "../connectors/ingestExternal";
import { searchLiveConnectors } from "../connectors/liveSearch";
import { FEED_BY_KEY, resolveActiveFeeds, RSS_POLL_INTERVAL_MS } from "../feeds/feedRegistry";
import type { CatalogFeed } from "../feeds/feedRegistry";
import { normalizeArticleBody } from "../extract/normalizeArticleBody";
import { assignClustersAsync } from "../dedup/clusters";
import { extractArticleFromUrl } from "../extract/readabilityExtract";
import { enqueueArticleImageFetch } from "../media/imageFetchQueue";
import { backfillMissingImages } from "../media/imageBackfill";
import { fetchRemoteText, probeFetchBackend } from "../fetch/remoteFetch";
import { getCachedRssXml } from "../fetch/rssCache";
import { parseRssXml } from "../rss/parser";
import { rankRssHeadlinesForQuery, rssItemToSearchArticle, buildSearchTerms, sanitizeAiKeywords } from "../search/relevance";
import { getEngineLibraryStats } from "./engineStats";
import { isDeepReadEnabled } from "./deepReadGate";
import { ensureSearchIndexLoaded, getSearchIndexSize, indexArticles, rankIndexedArticlesForQuery } from "../search/flexIndex";
import {
  db,
  getAllArticles,
  getAllClusters,
  getAllRssItems,
  getArticleCount,
  getCluster,
  getMultiSourceClusters,
  getPendingArticleCount,
  getMeta,
  getRssHeadlineCount,
  purgeBlockedNewsSources,
  getSummarizedCount,
  listPendingRssItems,
  setMeta,
  upsertArticle,
  upsertCluster,
  upsertRssItems,
} from "../storage/db";
import { expandQuery, getModelBootState, isSummarizerReady, subscribeModelBoot, summarizeArticle, waitForSummarizer } from "../summarize/summarizerClient";
import { needsEnglishDisplay } from "../summarize/languageDetect";
import { isFailedExtraction, normalizeSummarizerResult } from "../summarize/summaryQuality";
import { yieldToMain } from "../util/yieldToMain";
import type {
  ActivityEntry,
  ActivityKind,
  ArticleRecord,
  EngineLibraryStats,
  EngineStatus,
  FeedPollStatus,
  IntelBundle,
  LastSearchInfo,
  LastSummaryInfo,
  RssItem,
  SearchHit,
} from "../types";

export type SearchUpdate = {
  phase: "indexed" | "headlines" | "live" | "refined" | "done";
  hits: SearchHit[];
};

const EXPAND_QUERY_TIMEOUT_MS = 2_500;

async function expandQueryFast(query: string): Promise<string[]> {
  return Promise.race([
    expandQuery(query).catch(() => [] as string[]),
    new Promise<string[]>((resolve) => {
      setTimeout(() => resolve([]), EXPAND_QUERY_TIMEOUT_MS);
    }),
  ]);
}

function dedupeSearchHits(hits: SearchHit[]): SearchHit[] {
  const seen = new Set<string>();
  const sorted = [...hits].sort((a, b) => b.score - a.score);
  const out: SearchHit[] = [];
  for (const hit of sorted) {
    const key = hit.article.url || hit.article.id;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(hit);
  }
  return out;
}

async function loadClusters(ids: string[]): Promise<Map<string, Awaited<ReturnType<typeof getCluster>>>> {
  const unique = [...new Set(ids.filter(Boolean))];
  const pairs = await Promise.all(unique.map(async (id) => [id, await getCluster(id)] as const));
  return new Map(pairs);
}

async function buildHitsFromRanked(
  rankedArticles: { id: string; score: number }[],
  rankedHeadlines: { id: string; score: number }[],
  articles: ArticleRecord[],
  rssItems: RssItem[],
  liveHits: SearchHit[],
  clusterMap: Map<string, Awaited<ReturnType<typeof getCluster>>>,
): Promise<SearchHit[]> {
  const byId = new Map(articles.map((a) => [a.id, a]));
  const rssById = new Map(rssItems.map((r) => [r.id, r]));
  const results: SearchHit[] = [];
  const seenUrls = new Set<string>();

  for (const h of rankedArticles) {
    const article = byId.get(h.id);
    if (!article) continue;
    seenUrls.add(article.url);
    const cluster = article.clusterId ? (clusterMap.get(article.clusterId) ?? null) : null;
    results.push({ article, cluster, score: h.score, sourceKind: "indexed" });
  }

  for (const h of rankedHeadlines) {
    const item = rssById.get(h.id);
    if (!item) continue;
    if (seenUrls.has(item.link)) continue;
    seenUrls.add(item.link);
    const indexed = byId.get(h.id);
    const article = indexed ?? rssItemToSearchArticle(item);
    results.push({
      article,
      cluster: null,
      score: h.score,
      sourceKind: indexed ? "indexed" : "headline",
    });
  }

  for (const hit of liveHits) {
    if (seenUrls.has(hit.article.url)) continue;
    seenUrls.add(hit.article.url);
    results.push(hit);
  }

  return dedupeSearchHits(results);
}

type StatusListener = (s: EngineStatus) => void;

function activeFeeds(): CatalogFeed[] {
  return resolveActiveFeeds();
}

function buildInitialFeedStatuses(): FeedPollStatus[] {
  return activeFeeds().map((f) => ({
    key: f.key,
    label: f.label,
    state: "pending" as const,
  }));
}

let status: EngineStatus = {
  phase: "idle",
  message: "Ready",
  articlesIndexed: 0,
  rssHeadlines: 0,
  pendingArticles: 0,
  summarizedByModel: 0,
  feedsOk: 0,
  feedsFailed: 0,
  feedsTotal: buildInitialFeedStatuses().length,
  feedStatuses: buildInitialFeedStatuses(),
  lastPollAt: 0,
  modelReady: false,
  activityLog: [],
  lastSummary: null,
  lastSearch: null,
  clustersTotal: 0,
  multiSourceClusters: 0,
  connectorsIngested: 0,
  lastConnectorAt: 0,
  library: null,
  fetchMode: "",
  fetchProbeOk: false,
  fetchProbeDetail: "",
};

const listeners = new Set<StatusListener>();
let pollTimer: ReturnType<typeof setInterval> | null = null;
let processing = false;
const CLUSTER_WINDOW = 600;

function emit(patch: Partial<EngineStatus>) {
  status = { ...status, ...patch };
  listeners.forEach((l) => {
    try {
      l(status);
    } catch (err) {
      console.error("[GROVEE] status listener error", err);
    }
  });
}

function logActivity(kind: ActivityKind, message: string) {
  const entry: ActivityEntry = { ts: Date.now(), kind, message };
  const activityLog = [entry, ...status.activityLog].slice(0, 40);
  emit({ activityLog, message: entry.message });
}

async function syncDbStats() {
  const libraryBase = await getEngineLibraryStats(getSearchIndexSize());
  const library: EngineLibraryStats = {
    ...libraryBase,
    searchIndexSize: getSearchIndexSize(),
  };
  emit({
    articlesIndexed: library.articlesIndexed,
    rssHeadlines: library.rssHeadlines,
    pendingArticles: library.pendingArticles,
    summarizedByModel: library.summarizedByModel,
    library,
  });
}

export async function refreshEngineStats(): Promise<void> {
  await syncDbStats();
}

export function subscribeEngineStatus(fn: StatusListener): () => void {
  listeners.add(fn);
  fn(status);
  void syncDbStats();
  return () => listeners.delete(fn);
}

export function getEngineStatus(): EngineStatus {
  return status;
}

async function fetchFeedItems(feed: CatalogFeed): Promise<RssItem[]> {
  const urls = [feed.url, ...(feed.fallbackUrls ?? [])];
  let lastErr: unknown;

  const cached = await getCachedRssXml(urls);
  if (cached) {
    try {
      return parseRssXml(cached, {
        source: feed.label,
        sourceKey: feed.key,
        category: feed.category,
      });
    } catch (err) {
      lastErr = err;
    }
  }

  for (const url of urls) {
    try {
      const xml = await fetchRemoteText(url);
      return parseRssXml(xml, {
        source: feed.label,
        sourceKey: feed.key,
        category: feed.category,
      });
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("Feed fetch failed");
}

function setFeedStatus(
  feedStatuses: FeedPollStatus[],
  key: string,
  state: FeedPollStatus["state"],
  items?: number,
): FeedPollStatus[] {
  return feedStatuses.map((f) => (f.key === key ? { ...f, state, items: items ?? f.items } : f));
}

export async function pollAllFeeds(): Promise<void> {
  if (processing) {
    logActivity("rss", "Poll skipped — scan already running");
    return;
  }
  processing = true;
  const feeds = activeFeeds();

  try {
    emit({
      feedsTotal: feeds.length,
      feedStatuses: feeds.map((f) => {
        const prev = status.feedStatuses.find((s) => s.key === f.key);
        return prev ?? { key: f.key, label: f.label, state: "pending" as const };
      }),
    });

    logActivity("rss", `Starting RSS poll (${feeds.length} feeds)…`);
    emit({
      phase: "polling",
      message: `Scanning RSS 0/${feeds.length}…`,
      feedsOk: 0,
      feedsFailed: 0,
      feedStatuses: feeds.map((f) => ({ key: f.key, label: f.label, state: "pending" })),
    });

    let ok = 0;
    let failed = 0;
    let newItems = 0;
    const FEED_BATCH = 2;
    const INTER_BATCH_MS = 900;
    let batchFeedStatuses = feeds.map((f) => {
      const prev = status.feedStatuses.find((s) => s.key === f.key);
      return prev ?? { key: f.key, label: f.label, state: "pending" as const };
    });

    for (let i = 0; i < feeds.length; i += FEED_BATCH) {
      const chunk = feeds.slice(i, i + FEED_BATCH);
      const results = await Promise.allSettled(
        chunk.map(async (feed) => {
          const items = await fetchFeedItems(feed);
          const added = await upsertRssItems(items);
          return { feed, items, added };
        }),
      );

      for (let j = 0; j < results.length; j++) {
        const result = results[j];
        const feed = chunk[j];
        if (result.status === "fulfilled") {
          ok++;
          newItems += result.value.added;
          batchFeedStatuses = setFeedStatus(batchFeedStatuses, feed.key, "ok", result.value.items.length);
          logActivity("rss", `✓ ${feed.label} — ${result.value.items.length} headlines (${result.value.added} new)`);
        } else {
          failed++;
          batchFeedStatuses = setFeedStatus(batchFeedStatuses, feed.key, "fail");
          logActivity("error", `✗ ${feed.label} — fetch failed`);
        }
      }
      const done = Math.min(i + chunk.length, feeds.length);
      emit({
        feedStatuses: batchFeedStatuses,
        feedsOk: ok,
        feedsFailed: failed,
        message: `Scanning RSS ${done}/${feeds.length}…`,
      });
      if ((i / FEED_BATCH) % 4 === 3) await syncDbStats();
      await yieldToMain();
      if (i + FEED_BATCH < feeds.length) await new Promise((r) => setTimeout(r, INTER_BATCH_MS));
    }

    await setMeta("lastPollAt", String(Date.now()));
    logActivity("rss", `Poll complete: ${ok} OK, ${failed} failed, ${newItems} new headlines`);
    emit({
      feedsOk: ok,
      feedsFailed: failed,
      lastPollAt: Date.now(),
      phase: "extracting",
      message: newItems > 0 ? `Indexed ${newItems} new headlines` : ok > 0 ? "RSS scan done" : "RSS scan failed — check Log",
    });

    logActivity("connector", "Fetching GitHub repos + Hugging Face models…");
    if (isDeepReadEnabled()) {
      emit({ phase: "summarizing", message: "GitHub + Hugging Face Hub ingest…" });
      try {
        const ext = await ingestExternalSources({ maxGithub: 4, maxHf: 4 });
        const total = ext.github + ext.hf;
        logActivity(
          "connector",
          `✓ Connectors: ${ext.github} GitHub repos, ${ext.hf} HF models (${ext.skipped} skipped)`,
        );
        emit({
          connectorsIngested: status.connectorsIngested + total,
          lastConnectorAt: Date.now(),
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Connector ingest failed";
        logActivity("error", `Connectors: ${msg}`);
      }
    } else {
      logActivity("connector", "Connectors skipped (AI Deep Read off)");
    }

    await drainPendingArticles();
    void backfillMissingImages(5).catch(() => {});
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Poll failed";
    logActivity("error", `Poll aborted: ${msg}`);
    emit({ phase: "error", message: msg });
  } finally {
    processing = false;
    await syncDbStats();
  }
}

const LIVE_ROTATE_PAUSE_MS = 3000;
let rotateFeedIdx = 0;
let rotatePollBusy = false;

/** Poll one RSS feed in round-robin (Live tab rotation). Returns count of new headlines. */
export async function pollNextFeedInRotation(): Promise<number> {
  if (processing || rotatePollBusy) return 0;
  rotatePollBusy = true;
  const feeds = activeFeeds();
  if (!feeds.length) {
    rotatePollBusy = false;
    return 0;
  }

  const feed = feeds[rotateFeedIdx % feeds.length];
  rotateFeedIdx = (rotateFeedIdx + 1) % feeds.length;

  try {
    const items = await fetchFeedItems(feed);
    const added = await upsertRssItems(items);
    const prevStatuses = status.feedStatuses.length
      ? status.feedStatuses
      : feeds.map((f) => ({ key: f.key, label: f.label, state: "pending" as const }));

    emit({
      feedStatuses: setFeedStatus(prevStatuses, feed.key, "ok", items.length),
    });

    if (added > 0) {
      logActivity("rss", `↻ ${feed.label} — ${added} new headline${added === 1 ? "" : "s"}`);
      if (!isDeepReadEnabled()) {
        const ingested = await ingestHeadlineArticles(Math.min(added, 20));
        if (ingested.length) await indexArticles(ingested);
      }
      await syncDbStats();
    }
    return added;
  } catch {
    const prevStatuses = status.feedStatuses.length
      ? status.feedStatuses
      : feeds.map((f) => ({ key: f.key, label: f.label, state: "pending" as const }));
    emit({ feedStatuses: setFeedStatus(prevStatuses, feed.key, "fail") });
    return 0;
  } finally {
    rotatePollBusy = false;
  }
}

export { LIVE_ROTATE_PAUSE_MS };

async function ingestHeadlineArticles(max = 40): Promise<ArticleRecord[]> {
  const pending = await listPendingRssItems(max);
  if (!pending.length) return [];

  const drafts: ArticleRecord[] = pending.map((item) => {
    const base = rssItemToSearchArticle(item);
    const feedDef = FEED_BY_KEY[item.sourceKey];
    return {
      ...base,
      feedCategory: item.category,
      intelSource: "rss" as const,
      language: feedDef?.language ?? "multi",
      summarizedAt: 0,
    };
  });

  const ingested = await applyDisplayLanguageBatch(drafts);
  for (let i = 0; i < ingested.length; i++) {
    await upsertArticle(ingested[i]);
    if (i > 0 && i % 25 === 0) await yieldToMain();
  }
  return ingested;
}

async function drainPendingArticles(): Promise<void> {
  if (!isDeepReadEnabled()) {
    const ingested = await ingestHeadlineArticles(40);
    if (ingested.length) {
      await indexArticles(ingested);
    }
    const count = await getArticleCount();
    logActivity("index", `+${ingested.length} RSS headlines · ${count} searchable (Deep Read off)`);
    emit({ phase: "ready", articlesIndexed: count });
    return;
  }
  const batchSize = 8;
  const maxPerPoll = 48;
  let processed = 0;

  while (processed < maxPerPoll) {
    const pending = await listPendingRssItems(Math.min(batchSize, maxPerPoll - processed));
    if (!pending.length) break;
    await processArticleBatch(pending);
    processed += pending.length;
    await syncDbStats();
  }

  await rebuildIndex();
  const count = await getArticleCount();
  logActivity("index", `Index ready — ${count} articles searchable`);
  emit({ phase: "ready", articlesIndexed: count });
}

async function processArticleBatch(pending: RssItem[]): Promise<void> {
  logActivity("extract", `Processing batch of ${pending.length} articles…`);
  emit({ phase: "extracting" });

  for (const item of pending) {
    try {
      logActivity("extract", `Fetching page: ${item.title.slice(0, 50)}…`);
      let extracted = await extractArticleFromUrl(item.link, item.title);
      const rssText = item.description?.trim() ?? "";

      if (isFailedExtraction(extracted.title, extracted.text)) {
        logActivity("extract", `Blocked page — using RSS description: ${item.title.slice(0, 45)}…`);
        extracted = {
          title: item.title,
          text: rssText || extracted.text,
          image: item.image || extracted.image,
        };
      }

      emit({ phase: "summarizing" });
      logActivity("summarize", `Qwen summarizing: ${item.title.slice(0, 55)}…`);

      const bodyForSummary = normalizeArticleBody(extracted.text?.trim() || rssText, item.title).body;
      const rawSum = await summarizeArticle(bodyForSummary);
      const sum = normalizeSummarizerResult(
        rawSum,
        rssText,
        bodyForSummary,
        extracted.title || item.title,
      );
      const byModel = sum.summary.length > 20 && !sum.summary.startsWith(rssText.slice(0, 40));
      const feedDef = FEED_BY_KEY[item.sourceKey];
      let image = extracted.image || item.image;
      if (!image) {
        image = await enqueueArticleImageFetch(item.link).catch(() => "");
      }
      let article: ArticleRecord = {
        id: item.id,
        url: item.link,
        source: item.source,
        sourceKey: item.sourceKey,
        title: extracted.title || item.title,
        image,
        publishDate: item.published,
        publishedTs: item.publishedTs,
        articleText: bodyForSummary,
        summary: sum.summary,
        displayTitle: rawSum.titleEn,
        displaySummary: sum.summary,
        keyFacts: sum.keyFacts,
        keywords: [...sum.keywords, item.category, feedDef?.category ?? ""].filter(Boolean),
        entities: sum.entities,
        clusterId: item.id,
        confidence: "LOW",
        fetchedAt: Date.now(),
        summarizedAt: Date.now(),
        language: feedDef?.language ?? "multi",
        feedCategory: item.category,
        intelSource: "rss",
      };
      article = await ensureEnglishDisplay(article);
      await upsertArticle(article);

      const lastSummary: LastSummaryInfo = {
        title: article.displayTitle ?? article.title,
        source: article.source,
        summary: article.displaySummary ?? article.summary,
        keyFacts: article.keyFacts.slice(0, 4),
        byModel,
        at: Date.now(),
      };
      logActivity(
        "summarize",
        byModel ? `✓ Qwen summary saved (${article.source})` : `○ RSS fallback summary (${article.source})`,
      );
      emit({ lastSummary });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Processing failed";
      logActivity("error", `Failed: ${item.title.slice(0, 40)} — ${msg}`);
      let fallbackImage = item.image;
      if (!fallbackImage) {
        fallbackImage = await enqueueArticleImageFetch(item.link).catch(() => "");
      }
      let fallback: ArticleRecord = {
        id: item.id,
        url: item.link,
        source: item.source,
        sourceKey: item.sourceKey,
        title: item.title,
        image: fallbackImage,
        publishDate: item.published,
        publishedTs: item.publishedTs,
        articleText: item.description,
        summary: item.description.slice(0, 280),
        keyFacts: item.description ? [item.description.slice(0, 200)] : [],
        keywords: [],
        entities: [],
        clusterId: item.id,
        confidence: "LOW",
        fetchedAt: Date.now(),
        summarizedAt: 0,
        language: "multi",
        feedCategory: item.category,
        intelSource: "rss",
      };
      fallback = await ensureEnglishDisplay(fallback);
      await upsertArticle(fallback);
      emit({
        lastSummary: {
          title: fallback.displayTitle ?? fallback.title,
          source: fallback.source,
          summary: fallback.displaySummary ?? fallback.summary,
          keyFacts: fallback.keyFacts,
          byModel: false,
          at: Date.now(),
        },
      });
    }
  }
}

async function rebuildIndex(): Promise<void> {
  emit({ phase: "indexing" });
  logActivity("index", "Updating story clusters (recent window)…");
  const articles = await getAllArticles(CLUSTER_WINDOW);
  const { articles: clustered, clusters } = await assignClustersAsync(articles);
  await db.transaction("rw", db.articles, db.clusters, async () => {
    for (const a of clustered) await upsertArticle(a);
    for (const c of clusters) await upsertCluster(c);
  });
  await indexArticles(clustered);
  const multi = await getMultiSourceClusters();
  logActivity(
    "index",
    `Clustered ${clusters.length} groups · ${multi.length} cross-source stories detected`,
  );
  emit({ clustersTotal: clusters.length, multiSourceClusters: multi.length });
}

export async function hydrateEngineFromDb(): Promise<void> {
  logActivity("index", "Loading saved data from IndexedDB…");
  const lastPollRaw = await getMeta("lastPollAt");
  const lastPollAt = lastPollRaw ? Number(lastPollRaw) : 0;
  if (lastPollAt > 0) emit({ lastPollAt });

  await syncDbStats();

  const [articles, rssCount] = await Promise.all([getAllArticles(1500), getRssHeadlineCount()]);
  if (articles.length > 0) {
    await ensureSearchIndexLoaded(articles);
    const clusters = await getAllClusters();
    const multi = await getMultiSourceClusters();
    emit({
      phase: "ready",
      message: `${articles.length} articles · ${rssCount} RSS headlines in local DB`,
      clustersTotal: clusters.length,
      multiSourceClusters: multi.length,
    });
    logActivity("index", `Restored ${articles.length} articles · ${rssCount} headlines · ${multi.length} cross-source`);
    void (async () => {
      const rest = await getAllArticles(5000);
      if (rest.length > articles.length) {
        await ensureSearchIndexLoaded(rest);
      }
    })();
  } else if (rssCount > 0) {
    emit({
      phase: "ready",
      message: `${rssCount} RSS headlines in local DB — tap Refresh to update search index`,
    });
    logActivity("index", `Restored ${rssCount} RSS headlines from local storage`);
  } else {
    emit({ phase: "idle", message: "No cached headlines yet — tap Refresh feeds" });
  }
}

export async function searchNews(query: string, onUpdate?: (update: SearchUpdate) => void): Promise<SearchHit[]> {
  const q = query.trim();
  if (!q) return [];

  const emitUpdate = (phase: SearchUpdate["phase"], hits: SearchHit[]) => {
    onUpdate?.({ phase, hits });
  };

  logActivity("search", `Search: "${q}"`);
  emit({ message: `Searching: ${q}` });

  const [articles, rssItems] = await Promise.all([getAllArticles(), getAllRssItems()]);
  await yieldToMain();
  const totalHeadlines = rssItems.length;
  const indexedUrls = new Set(articles.map((a) => a.url));
  const byId = new Map(articles.map((a) => [a.id, a]));

  const livePromise = searchLiveConnectors(q);
  const expandPromise = expandQueryFast(q);

  const rankedArticles = await rankIndexedArticlesForQuery(articles, q, 24);
  const clusterMap = await loadClusters(
    rankedArticles.map((h) => byId.get(h.id)?.clusterId ?? "").filter(Boolean),
  );

  let hits = await buildHitsFromRanked(rankedArticles, [], articles, rssItems, [], clusterMap);
  emitUpdate("indexed", hits);
  await yieldToMain();

  const rankedHeadlines = rankRssHeadlinesForQuery(rssItems, q, indexedUrls, 40);
  hits = await buildHitsFromRanked(rankedArticles, rankedHeadlines, articles, rssItems, [], clusterMap);
  emitUpdate("headlines", hits);
  await yieldToMain();

  const live = await livePromise;
  hits = await buildHitsFromRanked(rankedArticles, rankedHeadlines, articles, rssItems, live.hits, clusterMap);
  emitUpdate("live", hits);

  const cleanedAi = sanitizeAiKeywords(await expandPromise);
  const terms = buildSearchTerms(q, cleanedAi);
  const rankOpts = { aiKeywords: cleanedAi };

  let refinedArticles = rankedArticles;
  let refinedHeadlines = rankedHeadlines;
  if (cleanedAi.length > 0) {
    await yieldToMain();
    refinedArticles = await rankIndexedArticlesForQuery(articles, q, 24, rankOpts);
    refinedHeadlines = rankRssHeadlinesForQuery(rssItems, q, indexedUrls, 40, rankOpts);
    hits = await buildHitsFromRanked(refinedArticles, refinedHeadlines, articles, rssItems, live.hits, clusterMap);
    emitUpdate("refined", hits);
  }

  const indexedMatches = refinedArticles.length;
  const headlineMatches = refinedHeadlines.length;
  const githubMatches = live.hits.filter((h) => h.sourceKind === "github").length;
  const hfMatches = live.hits.filter((h) => h.sourceKind === "huggingface").length;

  const lastSearch: LastSearchInfo = {
    query: q,
    expandedTerms: terms,
    resultCount: hits.length,
    indexedMatches,
    headlineMatches,
    githubMatches,
    hfMatches,
    liveGithubSkipped: live.githubSkipped,
    liveHfSkipped: live.hfSkipped,
    githubRateLimited: live.githubRateLimited,
    totalHeadlines,
    at: Date.now(),
  };

  logActivity(
    "search",
    live.githubSkipped && live.hfSkipped
      ? `Scanned ${totalHeadlines} headlines + ${articles.length} indexed (live APIs skipped — news query)`
      : `Scanned ${totalHeadlines} headlines + ${articles.length} indexed + live (${live.githubCount} GitHub, ${live.hfCount} HF${live.githubRateLimited ? ", GitHub rate-limited" : ""})`,
  );

  if (hits.length > 0) {
    logActivity(
      "search",
      `Found ${hits.length} (${indexedMatches} indexed + ${headlineMatches} headlines + ${githubMatches} GitHub + ${hfMatches} HF) — top: "${hits[0].article.title.slice(0, 45)}"`,
    );
  } else {
    logActivity("search", `No matches for "${q}"`);
  }

  emit({ phase: "ready", lastSearch });
  emitUpdate("done", hits);
  return hits;
}

export function buildIntelBundle(topic: string, hits: SearchHit[]): IntelBundle {
  const clusterIds = new Set(hits.map((h) => h.article.clusterId));
  const confidences = hits.map((h) => h.article.confidence);
  const confidence: IntelBundle["confidence"] = confidences.includes("HIGH")
    ? "HIGH"
    : confidences.includes("MEDIUM")
      ? "MEDIUM"
      : "LOW";

  return {
    topic,
    sources: clusterIds.size || hits.length,
    confidence,
    summaries: hits.map((h) => h.article.summary).filter(Boolean).slice(0, 8),
    keyFacts: [...new Set(hits.flatMap((h) => h.article.keyFacts))].slice(0, 12),
    articleLinks: hits.map((h) => h.article.url).slice(0, 12),
    images: hits.map((h) => h.article.image).filter(Boolean).slice(0, 8),
  };
}

export async function forcePollAllFeeds(): Promise<void> {
  if (processing) {
    logActivity("rss", "Resetting stuck scan lock…");
    processing = false;
  }
  await pollAllFeeds();
}

let engineBootPromise: Promise<void> | null = null;

export async function startEngine(): Promise<void> {
  if (!engineBootPromise) {
    engineBootPromise = bootEngine();
  }
  return engineBootPromise;
}

async function bootEngine(): Promise<void> {
  logActivity("index", "Engine boot…");
  let modelLogged = false;
  const syncModelReady = () => {
    if (!isDeepReadEnabled()) {
      emit({ modelReady: false });
      return;
    }
    const ready = isSummarizerReady() || getModelBootState().phase === "ready";
    emit({ modelReady: ready });
    if (ready && !modelLogged) {
      modelLogged = true;
      logActivity("model", "Qwen 2.5 0.5B ready for summaries");
    }
  };
  syncModelReady();
  subscribeModelBoot(() => syncModelReady());

  try {
    await hydrateEngineFromDb();
  } catch (err) {
    logActivity("error", `DB load failed: ${err instanceof Error ? err.message : "unknown"}`);
  }

  try {
    const probe = await probeFetchBackend();
    emit({
      fetchMode: probe.mode,
      fetchProbeOk: probe.ok,
      fetchProbeDetail: probe.detail,
    });
    logActivity(probe.ok ? "rss" : "error", `Fetch path (${probe.mode}): ${probe.detail}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Probe failed";
    emit({ fetchProbeOk: false, fetchProbeDetail: msg, fetchMode: "browser-relays" });
    logActivity("error", `Fetch probe: ${msg}`);
  }

  void pollAllFeeds();

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    void pollAllFeeds();
  }, RSS_POLL_INTERVAL_MS);

  void (async () => {
    try {
      const purged = await purgeBlockedNewsSources();
      if (purged.articles > 0 || purged.rss > 0) {
        logActivity(
          "rss",
          `Removed blocked outlets: ${purged.articles} articles, ${purged.rss} headlines`,
        );
        await rebuildIndex();
        await syncDbStats();
      }
    } catch (err) {
      logActivity("error", `Purge/reindex: ${err instanceof Error ? err.message : "failed"}`);
    }
  })();

  void (async () => {
    if (!isDeepReadEnabled()) return;
    if (!(await waitForSummarizer(180_000))) return;
    const all = await getAllArticles();
    let fixed = 0;
    for (const a of all) {
      if (fixed >= 15) break;
      if (a.displaySummary && a.displayTitle && !needsEnglishDisplay(a.displayTitle, a.displaySummary)) continue;
      if (!needsEnglishDisplay(a.title, a.summary)) continue;
      await upsertArticle(await ensureEnglishDisplay(a));
      fixed++;
    }
    if (fixed > 0) logActivity("index", `English display backfill: ${fixed} articles`);
  })();
}

export function stopEngine(): void {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}
