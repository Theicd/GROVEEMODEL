/**
 * Pre-LLM turn orchestration: web search, games, globe, and canned replies.
 * Used by local text models (SmolLM) without camera/attachments.
 */

import { isTriviaOrSocialGame } from "./gameSearch/gameIntents";
import { extractTriviaQuestionCount } from "./trivia/triviaPrompt";
import { isSimpleGreeting, isRtlText, type ChatTopic } from "./chatIntents";
import {
  buildLiveMediaInlineReply,
  filterUnifiedLiveMediaHits,
  isSportsLiveMediaRequest,
  liveMediaSerpHitsToUnified,
  resolveLiveMediaModeFromQuery,
  type InlineLiveMediaPayload,
} from "./chatInlineContent";
import { resolveEarlyTurnRouting, type UiLang } from "./chatRoutePrelude";
import type { ModelActivityEntry } from "./modelActivityLog";
import {
  buildGameSearchFoundReply,
  buildGameSearchNotFoundReply,
  parseGameUserRequest,
  searchOnlineGamesWithFallback,
  type GameCategoryId,
  type OnlineGame,
} from "./gameSearch";
import {
  buildGlobeCommand,
  shouldOpenGlobePanel,
} from "./realityGlobe/intents";
import type { GlobeCommand } from "./realityGlobe/bridge";
import {
  buildGlobeCommandFromSearch,
  shouldOpenGlobeForStructuredGeo,
} from "./realityGlobe/searchGlobeBridge";
import {
  buildGlobePlaceReply,
  buildPlacesMapReply,
  buildRouteMapReply,
} from "./realityGlobe/globePresentation";
import {
  buildNewsPanelGuideReply,
} from "./groveeNews/bridge";
import {
  buildUnifiedSearchPayload,
  type SearchResultsPayload,
} from "./searchResults";
import type { UnifiedSearchHit } from "./searchResults/types";
import {
  buildShortTimeReply,
  buildTimeWidgetFromStartupContext,
  buildTimeWidgetFromWorldTimeSource,
  isSinglePlaceTimeWidgetQuery,
} from "./timeWidget/resolveTimeWidget";
import type { TimeWidgetData } from "./timeWidget/types";
import type { StartupContext } from "./startupContext";
import {
  buildLocalTimeAnswer,
  isLocalContextTimeQuery,
} from "./startupContext";
import {
  runWebSearch,
  needsOpenWebEnrichment,
  wantsCinemaPlotSummaries,
  buildCapabilityLiveReply,
  buildWebFallbackNoDataReply,
  shouldDeliverStructuredLiveReply,
  isNewsQuery,
  isCurrencyQuery,
  isEarthquakeQuery,
  isDisasterQuery,
  isAviationQuery,
  isShipsQuery,
  isMarineInfraQuery,
  wantsNewsHeadlineBulletsInChat,
  type SearchIntent,
  type SearchSourceResult,
  type AnswerShape,
} from "./webSearch";
import { isStarlinkRegionalQuery } from "./webSearch/intents";
import { isCrossSourceQuery } from "./webSearch/crossSourceIntents";
import { shouldSearchLiveMedia } from "./webSearch/intents";
import {
  isLiveMediaCatalogQuery,
  isLiveTvCategoryChannelQuery,
  liveMediaCatalogSearchQuery,
} from "./liveMedia/mediaIntent";
import { fetchLiveMediaSearch, isRadioMediaQuery } from "./webSearch/providers/liveMediaSearch";
import type { SearchPlan } from "./webSearch/searchPlanner";
import type { SearchBrief } from "./webSearch";

export type PendingWebSearchMeta = {
  sources: SearchSourceResult[];
  summary: string;
  query?: string;
  answerShape?: AnswerShape;
  crossSource?: boolean;
};

export type StreamingSearchState = {
  sources: SearchSourceResult[];
  summary?: string;
  query?: string;
  brief?: SearchBrief;
  active: boolean;
} | null;

function beginChatSearchProgress(
  deps: ChatTurnPreludeDeps,
  query: string,
  summary: string,
) {
  deps.setStreamingSearchSources({
    sources: [],
    summary,
    query,
    active: true,
  });
}

function stashLiveMediaSearchMeta(
  deps: ChatTurnPreludeDeps,
  query: string,
  sources: SearchSourceResult[],
  summary: string,
) {
  deps.pendingWebSearchRef.current = {
    sources,
    summary,
    query,
  };
}

function deliverLiveMediaInlineCanned(
  deps: ChatTurnPreludeDeps,
  effectivePrompt: string,
  uiLang: UiLang,
  hits: UnifiedSearchHit[],
  mode: "livetv" | "radio",
  opts: { sportsPackage?: boolean; movies?: boolean },
  liveResultText: string,
) {
  const catalogLabel =
    mode === "radio" ? "מאגר רדיו" : opts.movies ? "ערוצי סרטים" : "TV LIVE";
  const sources: SearchSourceResult[] = [
    {
      provider: "live-tv",
      label: catalogLabel,
      ok: hits.length > 0,
      text: liveResultText,
      latencyMs: 0,
      error: hits.length ? undefined : "no channels",
    },
  ];
  const summary = hits.length ? `${hits.length} תוצאות` : "לא נמצאו ערוצים";
  stashLiveMediaSearchMeta(deps, effectivePrompt, sources, summary);

  if (hits.length) {
    deps.pendingInlineLiveMediaRef.current = {
      hits,
      mode,
      sportsPackage: opts.sportsPackage,
    };
    deps.deliverCanned(
      buildLiveMediaInlineReply(hits.length, uiLang, {
        sportsPackage: opts.sportsPackage,
        radio: mode === "radio",
        movies: opts.movies,
      }),
      liveResultText,
      "canned-live",
      mode === "radio" ? "Radio · inline" : "Live TV · inline",
    );
    return;
  }

  deps.deliverCanned(
    uiLang === "he"
      ? "לא מצאתי ערוצים בקטלוג המקומי. פתח TV LIVE מהתפריט ולחץ «סנכרון מקורות»."
      : "No channels found in the local catalog. Open TV LIVE from the menu and tap Sync sources.",
    "",
    "canned-live",
    "Live TV · inline",
  );
}

function finishChatSearchProgress(
  deps: ChatTurnPreludeDeps,
  query: string,
  sources: SearchSourceResult[],
  summary: string,
) {
  deps.setStreamingSearchSources({
    sources,
    summary,
    query,
    active: false,
  });
}

export type ChatTurnPreludeInput = {
  trimmed: string;
  effectivePrompt: string;
  priorTurns: ChatTurn[];
  chatTopic: ChatTopic;
  startupContext: StartupContext | null;
  desktopLayout: boolean;
  uiLang?: UiLang;
};

export type ChatTurnPreludeDeps = {
  setStatus: (s: string) => void;
  setStreamingSearchSources: (v: StreamingSearchState | ((prev: StreamingSearchState) => StreamingSearchState)) => void;
  setSearchResultsPayload: (v: SearchResultsPayload) => void;
  setSearchResultsOpen: (v: boolean) => void;
  setArtifactOpen: (v: boolean) => void;
  setGlobePanelOpen: (v: boolean) => void;
  setGlobeCommand: (v: GlobeCommand | null) => void;
  setGamesPanelOpen: (v: boolean) => void;
  setGamesPanelLayout: (v: "side" | "full") => void;
  setGamesPanelGames: (v: OnlineGame[]) => void;
  setGamesPanelTitle: (v: string) => void;
  setGamesPanelCategory: (v: GameCategoryId) => void;
  setGamesEmbedGame: (v: OnlineGame | null) => void;
  setStreamingGameCategoryPicker: (v: boolean) => void;
  pushActivity: (entry: Omit<ModelActivityEntry, "id" | "ts">) => void;
  resolveSearchPlan: (query: string, recentUserText: string[]) => Promise<SearchPlan | undefined>;
  qaForceLlm: () => boolean;
  qaHasPending: () => boolean;
  pendingWebSearchRef: { current: PendingWebSearchMeta | null };
  pendingTimeWidgetRef: { current: TimeWidgetData | null };
  pendingGameCategoryPickerRef: { current: boolean };
  pendingGameBrowseCategoryRef: { current: GameCategoryId | null };
  pendingInlineGamesRef: { current: OnlineGame[] | null };
  pendingInlineLiveMediaRef: { current: InlineLiveMediaPayload | null };
  deliverCanned: (reply: string, webContext: string, replySource: string, activityTitle?: string) => void;
  onResetSession?: () => void;
};

export type ChatTurnPreludeContinue = {
  webContext: string;
  searchHint: string;
  gameSearchHint: string;
  gameGroundingBlock: string;
  gameNoResults: boolean;
  globePlaceLabel: string;
  shouldRunWebSearch: boolean;
  localTimeOnly: boolean;
  greeting: boolean;
  triviaMode: boolean;
  triviaQuestionCount: number;
};

export type ChatTurnPreludeOutcome =
  | { action: "canned" }
  | { action: "continue"; ctx: ChatTurnPreludeContinue };

/** Search, games, globe panels, and structured canned replies — no camera/attachments. */
export async function runTextChatTurnPrelude(
  input: ChatTurnPreludeInput,
  deps: ChatTurnPreludeDeps,
): Promise<ChatTurnPreludeOutcome> {
  const { trimmed, effectivePrompt, priorTurns, startupContext, desktopLayout } = input;
  let chatTopic = input.chatTopic;
  const uiLang: UiLang =
    input.uiLang ?? (isRtlText(trimmed || effectivePrompt) ? "he" : "en");
  const greeting = isSimpleGreeting(effectivePrompt);

  const earlyRoute = resolveEarlyTurnRouting({
    text: trimmed,
    effectivePrompt,
    chatTopic,
    uiLang,
    startupContext,
    blockGames: false,
  });
  if (earlyRoute.action === "canned") {
    if (earlyRoute.resetSession) deps.onResetSession?.();
    deps.deliverCanned(
      earlyRoute.reply,
      "",
      earlyRoute.replySource,
      earlyRoute.replySource === "reset" ? "איפוס שיחה" : "GROVEE · routing",
    );
    return { action: "canned" };
  }

  chatTopic = earlyRoute.chatTopic;
  let wantsGameSearch = earlyRoute.wantsGameSearch;
  let shouldRunWebSearch = earlyRoute.shouldRunWebSearch;

  if (greeting) {
    wantsGameSearch = false;
    shouldRunWebSearch = false;
  }

  const liveCatalogQuery = isLiveMediaCatalogQuery(effectivePrompt);
  if (liveCatalogQuery) {
    wantsGameSearch = false;
    shouldRunWebSearch = false;
  }

  deps.pendingInlineGamesRef.current = null;
  deps.pendingInlineLiveMediaRef.current = null;

  let webContext = "";
  let searchHint = "";
  let marineLiveCannedReply: string | null = null;
  deps.pendingWebSearchRef.current = null;
  deps.pendingTimeWidgetRef.current = null;

  const localTimeOnly =
    !wantsGameSearch &&
    !!startupContext &&
    isLocalContextTimeQuery(trimmed || effectivePrompt);

  let searchIntentsForGlobe: SearchIntent[] = [];
  let lastSearchSources: SearchSourceResult[] = [];

  const deliverLiveCannedReply = (
    reply: string,
    ctx: string,
    replySource: string,
    title = "Live data · canned reply",
  ): boolean => {
    const text = reply.trim();
    if (!text) return false;
    if (deps.qaForceLlm()) return false;
    if (wantsGameSearch) return false;
    deps.deliverCanned(text, ctx, replySource, title);
    return true;
  };

  const finishCannedLive = (reply: string, ctx: string, replySource: string): boolean => {
    if (!deps.qaHasPending()) return false;
    return deliverLiveCannedReply(reply, ctx, replySource);
  };

  if (!shouldRunWebSearch && !localTimeOnly) {
    const preCanned = buildCapabilityLiveReply(effectivePrompt, [], []);
    if (preCanned && isStarlinkRegionalQuery(effectivePrompt)) {
      if (finishCannedLive(preCanned, "", "canned-live")) return { action: "canned" };
    }
  }

  if (localTimeOnly && startupContext) {
    deps.pendingTimeWidgetRef.current = buildTimeWidgetFromStartupContext(startupContext);
    webContext = buildLocalTimeAnswer(startupContext, effectivePrompt);
    searchHint = " · זמן מקומי (ללא חיפוש ברשת)";
    deps.pushActivity({
      direction: "system",
      kind: "web_search",
      title: "Local Context",
      detail: webContext,
    });
  } else if (shouldRunWebSearch) {
    deps.setStatus("מחפש מידע…");
    deps.setStreamingSearchSources({
      sources: [],
      summary: "מנתב שאילתה…",
      query: effectivePrompt,
      active: true,
    });
    try {
      const recentUserText = priorTurns
        .filter((t) => t.role === "user")
        .slice(-4)
        .map((t) => t.content);
      const searchPlan = await deps.resolveSearchPlan(effectivePrompt, recentUserText);
      const planHint = searchPlan
        ? {
            queries: searchPlan.queries,
            answerShape: searchPlan.answerShape,
            useWebFallback: searchPlan.useWebFallback ?? true,
            blendNewsWithWeb: searchPlan.blendNewsWithWeb ?? false,
          }
        : undefined;
      const searchResult = await runWebSearch(effectivePrompt, {
        recentUserText,
        plan: planHint,
        onProgress: (ev) => {
          if (ev.type === "provider_done") {
            deps.setStreamingSearchSources((prev) => {
              if (!prev) return prev;
              const rest = prev.sources.filter((s) => s.provider !== ev.result.provider);
              return { ...prev, sources: [...rest, ev.result] };
            });
          }
          if (ev.type === "complete") {
            deps.setStreamingSearchSources((prev) =>
              prev ? { ...prev, sources: ev.sources, active: false } : null,
            );
          }
        },
      });
      searchIntentsForGlobe = searchResult.intents;
      lastSearchSources = searchResult.sources;
      webContext = searchResult.contextText;
      const unifiedSearchPayload = buildUnifiedSearchPayload(effectivePrompt, searchResult.sources);
      const newsPayloadAfterSearch = unifiedSearchPayload.hits.length
        ? { cardCount: unifiedSearchPayload.facets.rss, mode: "search" as const }
        : null;
      if (isSinglePlaceTimeWidgetQuery(effectivePrompt)) {
        const wt = searchResult.sources.find((s) => s.provider === "world-time" && s.ok);
        const widget = wt ? buildTimeWidgetFromWorldTimeSource(wt) : null;
        if (widget) deps.pendingTimeWidgetRef.current = widget;
      }
      const searchLiveOk = searchResult.sources.some((s) => s.ok && s.text.trim());
      marineLiveCannedReply =
        searchResult.cannedReply ??
        buildCapabilityLiveReply(effectivePrompt, searchResult.intents, searchResult.sources, {
          answerShape: searchPlan?.answerShape,
        });
      if (
        !marineLiveCannedReply &&
        shouldDeliverStructuredLiveReply(
          effectivePrompt,
          searchResult.intents,
          searchResult.sources,
          null,
        )
      ) {
        marineLiveCannedReply = buildCapabilityLiveReply(
          effectivePrompt,
          searchResult.intents,
          searchResult.sources,
          { answerShape: searchPlan?.answerShape },
        );
      }
      const newsQueryTurn = isNewsQuery(effectivePrompt);
      const newsHeadlineBulletsTurn = wantsNewsHeadlineBulletsInChat(effectivePrompt);
      const qaForceLlm = deps.qaForceLlm();
      const shouldDeliverLive =
        !qaForceLlm &&
        !!marineLiveCannedReply &&
        !wantsGameSearch &&
        (!newsQueryTurn || newsHeadlineBulletsTurn) &&
        (searchLiveOk || !searchResult.sources.some((s) => s.ok && s.provider !== "searxng")) &&
        shouldDeliverStructuredLiveReply(
          effectivePrompt,
          searchResult.intents,
          searchResult.sources,
          marineLiveCannedReply,
        );
      if (
        !searchLiveOk &&
        !marineLiveCannedReply &&
        (searchPlan?.useWebFallback || searchResult.sources.some((s) => s.provider === "searxng"))
      ) {
        marineLiveCannedReply = buildWebFallbackNoDataReply(effectivePrompt, searchResult.sources);
      }
      deps.pendingWebSearchRef.current = {
        sources: searchResult.sources,
        summary: searchResult.summaryHe,
        query: effectivePrompt,
        answerShape: searchPlan?.answerShape,
        crossSource:
          isCrossSourceQuery(effectivePrompt) ||
          searchResult.intents.length >= 2 ||
          !!searchPlan?.blendNewsWithWeb,
      };
      const liveTvSource = searchResult.sources.find(
        (s) => s.provider === "live-tv" && s.ok && s.liveMediaHits?.length,
      );
      if (liveTvSource?.liveMediaHits?.length) {
        const allHits = liveMediaSerpHitsToUnified(liveTvSource.liveMediaHits);
        const mode = resolveLiveMediaModeFromQuery(effectivePrompt, allHits);
        const hits = filterUnifiedLiveMediaHits(allHits, mode);
        if (hits.length) {
          deps.pendingInlineLiveMediaRef.current = {
            hits,
            mode,
            sportsPackage: isSportsLiveMediaRequest(effectivePrompt),
          };
        }
      }
      if (liveCatalogQuery && deps.pendingInlineLiveMediaRef.current?.hits.length) {
        marineLiveCannedReply = null;
        deps.pendingWebSearchRef.current = null;
      }
      deps.setStreamingSearchSources({
        sources: searchResult.sources,
        summary: searchResult.summaryHe,
        query: effectivePrompt,
        brief: searchResult.brief,
        active: false,
      });
      if (
        shouldDeliverLive &&
        marineLiveCannedReply &&
        deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live")
      ) {
        return { action: "canned" };
      }
      if (
        needsOpenWebEnrichment(effectivePrompt) &&
        !wantsCinemaPlotSummaries(effectivePrompt) &&
        marineLiveCannedReply &&
        !wantsGameSearch &&
        !deps.qaForceLlm() &&
        deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live", "Open web · canned reply")
      ) {
        return { action: "canned" };
      }
      if (
        newsQueryTurn &&
        !newsHeadlineBulletsTurn &&
        !wantsGameSearch &&
        !deps.qaForceLlm()
      ) {
        const guide = buildNewsPanelGuideReply(
          effectivePrompt,
          newsPayloadAfterSearch
            ? { mode: newsPayloadAfterSearch.mode, cardCount: newsPayloadAfterSearch.cardCount }
            : null,
        );
        if (deliverLiveCannedReply(guide, "", "canned-live", "GROVEE NEWS · פאנל")) {
          return { action: "canned" };
        }
      }
      if (!webContext.trim()) {
        searchHint = " · אין תוצאות חיפוש";
      } else {
        searchHint = unifiedSearchPayload.hits.length
          ? ` · ${unifiedSearchPayload.hits.length} מקורות · ${searchResult.summaryHe}`
          : ` · ${searchResult.summaryHe}`;
      }
      deps.pushActivity({
        direction: "system",
        kind: "web_search",
        title: "Web Search",
        detail: [
          searchResult.summaryHe,
          `intents: ${searchResult.intents.join(", ")}`,
          ...searchResult.sources.map(
            (s) => `${s.label}: ${s.ok ? "OK" : "FAIL"}${s.error ? ` (${s.error})` : ""} · ${s.latencyMs}ms`,
          ),
        ].join("\n"),
      });
    } catch {
      webContext = "";
      searchHint = " · חיפוש נכשל";
      deps.pendingWebSearchRef.current = null;
      deps.setStreamingSearchSources(null);
    }
  }

  let globePlaceCannedReply: string | null = null;
  let globePlaceLabel = "";
  const globeFromSearch = lastSearchSources.length
    ? buildGlobeCommandFromSearch(trimmed || effectivePrompt, searchIntentsForGlobe, lastSearchSources)
    : null;
  const openGlobe =
    !wantsGameSearch &&
    desktopLayout &&
    (globeFromSearch != null ||
      shouldOpenGlobeForStructuredGeo(trimmed || effectivePrompt, searchIntentsForGlobe, lastSearchSources) ||
      shouldOpenGlobePanel(trimmed || effectivePrompt, searchIntentsForGlobe));
  if (!openGlobe) {
    deps.setGlobePanelOpen(false);
    deps.setGlobeCommand(null);
  }
  if (openGlobe) {
    const cmd = globeFromSearch ?? buildGlobeCommand(trimmed || effectivePrompt, searchIntentsForGlobe);
    if (cmd) {
      deps.setGlobePanelOpen(true);
      deps.setGlobeCommand(cmd);
      deps.setArtifactOpen(false);
      deps.setGamesPanelOpen(false);
      deps.setSearchResultsOpen(false);
      if (cmd.type === "flyTo" && cmd.label) {
        const places = lastSearchSources.find((s) => s.provider === "nominatim-places" && s.ok);
        globePlaceLabel = cmd.label;
        globePlaceCannedReply = buildPlacesMapReply(cmd.label, places?.url);
      } else if (cmd.type === "drawRoute") {
        const dist = lastSearchSources.find((s) => s.provider === "osrm-distance" && s.ok);
        const lines = dist?.text.split("\n") ?? [];
        const fromLine = lines.find((l) => l.startsWith("מ:"));
        const toLine = lines.find((l) => l.startsWith("אל:"));
        const kmLine = lines.find((l) => l.includes('ק"מ'));
        const timeLine = lines.find((l) => l.startsWith("זמן"));
        globePlaceCannedReply = buildRouteMapReply(
          fromLine?.replace(/^מ:\s*/, "").split("(")[0]?.trim() ?? "מקור",
          toLine?.replace(/^אל:\s*/, "").split("(")[0]?.trim() ?? "יעד",
          kmLine?.match(/([\d.]+)\s*ק"מ/)?.[1],
          timeLine?.replace("זמן נסיעה משוער:", "").trim(),
        );
      } else if (cmd.type === "focusPlaceQuiet" && cmd.presentation !== false) {
        globePlaceLabel = cmd.name;
        globePlaceCannedReply = buildGlobePlaceReply(cmd.name);
      }
    }
  }

  let gameSearchHint = "";
  deps.pendingGameCategoryPickerRef.current = false;
  deps.pendingGameBrowseCategoryRef.current = null;
  deps.setStreamingGameCategoryPicker(false);
  let gameGroundingBlock = "";
  let gameNoResults = false;
  let gameSearchCannedReply: string | null = null;
  let inlineGames: OnlineGame[] = [];

  if (wantsGameSearch) {
    deps.setStreamingSearchSources(null);
    beginChatSearchProgress(deps, effectivePrompt, "מחפש משחקים…");
    deps.setStatus("מחפש משחקים…");
    try {
      const gameReq = parseGameUserRequest(trimmed || effectivePrompt);
      const panelCategory = gameReq.category ?? "featured";
      const gameResult = await searchOnlineGamesWithFallback(gameReq, 12);
      deps.setGamesPanelCategory(panelCategory);
      deps.setGamesPanelGames(gameResult.games);
      deps.setGamesPanelTitle(gameReq.panelTitle);
      deps.setGamesEmbedGame(null);

      if (gameResult.matchFound && gameResult.games.length) {
        inlineGames = gameResult.games;
        deps.pendingInlineGamesRef.current = gameResult.games;
        gameSearchHint = ` · ${gameResult.games.length} משחקים`;
        gameGroundingBlock = gameResult.games.map((g, i) => `${i + 1}. ${g.title}`).join("\n");
        gameSearchCannedReply = buildGameSearchFoundReply(gameResult.games.length, gameReq);
        finishChatSearchProgress(
          deps,
          effectivePrompt,
          [
            {
              provider: "live-tv",
              label: "ארכיון משחקים",
              ok: true,
              text: `${gameResult.games.length} משחקים`,
              latencyMs: gameResult.latencyMs,
            },
          ],
          `${gameResult.games.length} משחקים נמצאו`,
        );
        deps.pushActivity({
          direction: "system",
          kind: "game_search",
          title: "Game Search",
          detail: [
            gameReq.query || gameReq.panelTitle || "(browse)",
            `category: ${panelCategory}`,
            `count: ${gameResult.games.length}`,
            `latency: ${gameResult.latencyMs}ms`,
            gameReq.yearFrom != null ? `years: ${gameReq.yearFrom}-${gameReq.yearTo}` : "",
            ...gameResult.games.slice(0, 5).map((g) => g.title),
          ]
            .filter(Boolean)
            .join("\n"),
        });
      } else {
        gameNoResults = true;
        deps.setGamesPanelGames([]);
        deps.setGamesPanelTitle(gameReq.panelTitle);
        deps.pendingGameCategoryPickerRef.current = true;
        deps.pendingGameBrowseCategoryRef.current = panelCategory;
        deps.setStreamingGameCategoryPicker(true);
        gameSearchHint = " · לא נמצא — קטגוריות";
        gameSearchCannedReply = buildGameSearchNotFoundReply(gameReq);
        finishChatSearchProgress(
          deps,
          effectivePrompt,
          [{ provider: "live-tv", label: "ארכיון משחקים", ok: false, text: "", error: "no match", latencyMs: 0 }],
          "לא נמצאו משחקים",
        );
        deps.pushActivity({
          direction: "system",
          kind: "game_search",
          title: "Game Search · no match",
          detail: [
            gameReq.query || gameReq.panelTitle || "(browse)",
            `category: ${panelCategory}`,
            "matchFound: false",
          ].join("\n"),
        });
      }
    } catch {
      gameSearchHint = " · חיפוש משחקים נכשל";
      gameNoResults = true;
      deps.pendingGameCategoryPickerRef.current = true;
      deps.setStreamingGameCategoryPicker(true);
      deps.setGamesPanelGames([]);
      gameSearchCannedReply = buildGameSearchNotFoundReply(
        parseGameUserRequest(trimmed || effectivePrompt),
      );
    }
  }

  const qaForceLlm = deps.qaForceLlm();

  if (!qaForceLlm && wantsGameSearch && gameSearchCannedReply) {
    if (inlineGames.length) deps.pendingInlineGamesRef.current = inlineGames;
    deps.deliverCanned(gameSearchCannedReply, "", "canned-game");
    return { action: "canned" };
  }

  const wantsStandaloneLiveMedia =
    !greeting &&
    !wantsGameSearch &&
    !shouldRunWebSearch &&
    (shouldSearchLiveMedia(effectivePrompt, false) || liveCatalogQuery);

  if (wantsStandaloneLiveMedia) {
    const sportsPackage = isSportsLiveMediaRequest(effectivePrompt);
    const moviesCategory = isLiveTvCategoryChannelQuery(effectivePrompt) && /סרט|movies?/i.test(effectivePrompt);
    const liveQuery = sportsPackage
      ? "sport"
      : liveCatalogQuery
        ? liveMediaCatalogSearchQuery(effectivePrompt)
        : effectivePrompt;
    const countryCode = startupContext?.countryCode;
    const progressLabel = isRadioMediaQuery(liveQuery)
      ? "מחפש תחנות רדיו…"
      : moviesCategory
        ? "מחפש ערוצי סרטים…"
        : "מחפש ערוצים חיים…";
    beginChatSearchProgress(deps, effectivePrompt, progressLabel);
    deps.setStatus(progressLabel);
    try {
      const liveResult = await fetchLiveMediaSearch(liveQuery, {
        panelSearch: true,
        countryCode,
        catalogSearch: liveCatalogQuery,
      });
      const allHits = liveMediaSerpHitsToUnified(liveResult.liveMediaHits ?? []);
      const mode = resolveLiveMediaModeFromQuery(liveQuery, allHits);
      const hits = filterUnifiedLiveMediaHits(allHits, mode);
      if (!deps.qaForceLlm()) {
        finishChatSearchProgress(
          deps,
          effectivePrompt,
          [
            {
              provider: "live-tv",
              label:
                mode === "radio" ? "מאגר רדיו" : moviesCategory ? "ערוצי סרטים" : "TV LIVE",
              ok: hits.length > 0,
              text: liveResult.text,
              latencyMs: liveResult.latencyMs,
              error: hits.length ? undefined : liveResult.error ?? "no channels",
            },
          ],
          hits.length ? `${hits.length} תוצאות` : "לא נמצאו ערוצים",
        );
        deliverLiveMediaInlineCanned(
          deps,
          effectivePrompt,
          uiLang,
          hits,
          mode,
          { sportsPackage, movies: moviesCategory },
          liveResult.text,
        );
        return { action: "canned" };
      }
    } catch {
      deps.setStreamingSearchSources(null);
      if (!deps.qaForceLlm()) {
        deliverLiveMediaInlineCanned(
          deps,
          effectivePrompt,
          uiLang,
          [],
          "livetv",
          { sportsPackage, movies: moviesCategory },
          "",
        );
        return { action: "canned" };
      }
    }
  }

  if (!qaForceLlm && globePlaceCannedReply) {
    deps.deliverCanned(globePlaceCannedReply, webContext, "canned-globe");
    deps.pushActivity({
      direction: "system",
      kind: "globe_focus",
      title: "Globe · place focus",
      detail: globePlaceLabel || globePlaceCannedReply,
    });
    return { action: "canned" };
  }

  if (
    !qaForceLlm &&
    deps.pendingTimeWidgetRef.current &&
    isSinglePlaceTimeWidgetQuery(effectivePrompt)
  ) {
    const reply = buildShortTimeReply(deps.pendingTimeWidgetRef.current);
    deps.deliverCanned(reply, "", "local-time");
    deps.pushActivity({
      direction: "system",
      kind: "web_search",
      title: "Time widget",
      detail: reply,
    });
    return { action: "canned" };
  }

  const structuredLiveTurn =
    !qaForceLlm &&
    !isNewsQuery(effectivePrompt) &&
    !wantsGameSearch &&
    !!marineLiveCannedReply &&
    shouldDeliverStructuredLiveReply(
      effectivePrompt,
      searchIntentsForGlobe,
      lastSearchSources,
      marineLiveCannedReply,
    );
  if (structuredLiveTurn && marineLiveCannedReply && !globePlaceCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live")) {
      return { action: "canned" };
    }
  }

  const pureCurrencyTurn =
    !qaForceLlm &&
    !wantsGameSearch &&
    isCurrencyQuery(effectivePrompt) &&
    !!marineLiveCannedReply;
  if (pureCurrencyTurn && marineLiveCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live", "Frankfurter · canned reply")) {
      return { action: "canned" };
    }
  }

  const pureEarthquakeTurn =
    !qaForceLlm &&
    !wantsGameSearch &&
    isEarthquakeQuery(effectivePrompt) &&
    lastSearchSources.some((s) => s.provider === "usgs-earthquake" && s.ok && s.text.trim()) &&
    !!marineLiveCannedReply;
  if (pureEarthquakeTurn && marineLiveCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live", "USGS · canned reply")) {
      return { action: "canned" };
    }
  }

  const pureDisasterTurn =
    !qaForceLlm &&
    !wantsGameSearch &&
    isDisasterQuery(effectivePrompt) &&
    !isEarthquakeQuery(effectivePrompt) &&
    lastSearchSources.some((s) => s.provider === "gdacs-disasters" && s.ok && s.text.trim()) &&
    !!marineLiveCannedReply;
  if (pureDisasterTurn && marineLiveCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live", "GDACS · canned reply")) {
      return { action: "canned" };
    }
  }

  const pureAviationTurn =
    !qaForceLlm &&
    !wantsGameSearch &&
    (isAviationQuery(effectivePrompt) || /כמה\s+מטוס/i.test(effectivePrompt)) &&
    lastSearchSources.some((s) => s.provider === "adsb-aviation" && s.ok && s.text.trim()) &&
    !!marineLiveCannedReply;
  if (pureAviationTurn && marineLiveCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live", "ADS-B · canned reply")) {
      return { action: "canned" };
    }
  }

  const pureShipsTurn =
    !qaForceLlm &&
    !wantsGameSearch &&
    isShipsQuery(effectivePrompt) &&
    !isMarineInfraQuery(effectivePrompt) &&
    lastSearchSources.some((s) => s.provider === "ais-ships" && s.ok && s.text.trim()) &&
    !!marineLiveCannedReply;
  if (pureShipsTurn && marineLiveCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live", "AIS · canned reply")) {
      return { action: "canned" };
    }
  }

  if (deps.qaHasPending() && structuredLiveTurn && marineLiveCannedReply) {
    if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "canned-live")) {
      return { action: "canned" };
    }
  }

  return {
    action: "continue",
    ctx: {
      webContext,
      searchHint,
      gameSearchHint,
      gameGroundingBlock,
      gameNoResults,
      globePlaceLabel,
      shouldRunWebSearch,
      localTimeOnly,
      greeting,
      triviaMode: isTriviaOrSocialGame(trimmed) || chatTopic === "trivia",
      triviaQuestionCount: extractTriviaQuestionCount(trimmed),
    },
  };
}