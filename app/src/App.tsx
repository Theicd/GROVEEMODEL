import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  CODE_TOKEN_CAP,
  CODE_TOKEN_FLOOR,
  CONTINUE_CODE_SYSTEM_HINT,
  isCodeGenerationRequest,
  isRtlText,
  isSimpleGreeting,
  isPersonActivityQuestion,
  isPersonVisibilityQuestion,
  isCurrentPersonStateQuestion,
  isFingerCountQuestion,
  needsPersonFocusRefresh,
  isSceneInterpretationQuestion,
  isVisualDetailQuestion,
  needsLiveCameraContext,
  isConsciousnessQuestion,
  isPersonDemographicsQuestion,
  isPersonMoodQuestion,
  isConversationFirstRequest,
  formatCameraTopicLabel,
  needsVisionSensorContext,
  needsAttachedDocumentAnalysis,
  wantsExactTextExtraction,
  isVisionUnrelatedTurn,
  shouldContinueCode,
  getArtifactScanContent,
  splitAssistantStream,
  trimHistoryForContext,
  classifyChatTopic,
  isTopicShift,
  topicShiftHint,
  type ChatTopic,
  type ChatTurn,
} from "./chatIntents";
import {
  defaultVisionPrompt,
  MAX_ATTACHMENTS,
  type StoredMessageImage,
} from "./imageAttachments";
import {
  attachmentKindLabel,
  buildIngestedDocumentPromptBlock,
  DOCUMENT_ACCEPT,
  hasSubstantialExtractedText,
  ingestDocument,
  isAcceptedDocumentFile,
  revokePendingAttachment,
  type PendingAttachment,
} from "./documentIngest";
import { IntroScreen } from "./components/IntroScreen";
import { GroveeInfoModal } from "./components/GroveeInfoModal";
import { GroveeLogoMark } from "./GroveeLogoMark";
import { ChatMessageAvatar } from "./ChatMessageAvatar";
import { SidebarGearMenu, type SidebarGearAction } from "./SidebarGearMenu";
import { GlobeVisual } from "./GlobeVisual";
import { ChatMarkdown } from "./chatMarkdown";
import { ArtifactPanel, type Artifact } from "./ArtifactPanel";
import {
  buildPersistedAssistantPayload,
  extractPrimaryArtifact,
  extractRichParts,
} from "./artifacts";
import { downloadProgressPercent } from "./introProgressFormat";
import { formatBytes, requestPersistentStorage } from "./storageReport";
import {
  SCENE_ANALYSIS_SYSTEM_PROMPT,
  buildSceneAnalysisUserPrompt,
} from "./cameraPrompts";
import {
  CAMERA_ANTI_DEFLECT_APPEND,
  CHARACTER_ACTIVITY_APPEND,
  CHARACTER_INTERPRETATION_APPEND,
  CHARACTER_MODE_CHAT_APPEND,
  CURRENT_PERSON_STATE_APPEND,
  GREETING_CAMERA_STARTING_APPEND,
  GREETING_WITH_CAMERA_APPEND,
  PERSON_VISIBILITY_CHAT_APPEND,
  PROACTIVE_UTTERANCE_SYSTEM,
  buildProactiveUserPrompt,
  TOPIC_SHIFT_CHAT_APPEND,
  VISION_ESCALATION_CHAT_APPEND,
  FINGER_COUNT_CHAT_APPEND,
  HOLDING_CHAT_APPEND,
  MOOD_CHAT_APPEND,
  CAMERA_CONVERSATION_APPEND,
  CAMERA_PURE_CHAT_APPEND,
  DOCUMENT_IMAGE_CHAT_APPEND,
  CAMERA_HAL_SYSTEM,
  GROVEE_CHAT_SYSTEM,
  migrateGemmaSystemPrompt,
  buildLanguageReplyDirective,
  buildWebSearchGroundingAppend,
  WEB_SEARCH_NO_RESULTS_APPEND,
  GAME_SEARCH_GROUNDING_APPEND,
  GAME_SEARCH_NO_RESULTS_APPEND,
} from "./characterPrompts";
import { extractTextFromDocumentImages } from "./documentOcr";
import { buildProactiveSensorBlock, poseFromWorld } from "./sensorBlock";
import { buildFingerCountBlock } from "./visionBridge";
import {
  buildDeepVisionContextBlock,
  buildFingerAnswerBlock,
} from "./vision2/dialogueContext";
import {
  buildChatVisionContextBlock,
  buildInternalVisionContextForUi,
  CHAT_VISION_SYSTEM_HINT,
} from "./vision2/liveVisionChatBrief";
import type { HalMoodState } from "./vision2/halMoodEngine";
import { formatFreshPersonBlock } from "./personFocus";
import {
  attachStreamToVideo,
  requestCameraStream,
  type CameraStreamHandle,
} from "./cameraMode";
import { CameraUserProfilePanel } from "./CameraUserProfilePanel";
import {
  appendTopicToLog,
  buildCameraHistoryForWorker,
  clearCameraSessionStore,
  loadCameraSessionStore,
  saveCameraSessionStore,
  type CameraMessage,
  type CameraSessionStore,
  type UserProfile,
} from "./cameraSession";
import {
  buildUserMemoryPromptBlock,
  findRelevantHistoryForPrompt,
  patchCameraStoreAfterTurn,
  searchCameraHistory,
  updateUserProfile,
  buildRollingSummary,
} from "./cameraUserMemory";
import { checkBrowserVisionSupport } from "./browserVision";
import { detectVisionBudget } from "./visionBudget";
import { CharacterBrain, moodLabelHe, type CharacterMood } from "./characterBrain";
import { WorldMemory } from "./worldMemory";
import {
  GroveeVisionRunner,
  type CharacterDecision,
  type SceneAnalysisResult,
} from "./GroveeVisionRunner";
import { mountGroveeVisionProbe } from "./visionQaProbe";
import { CameraPreview } from "./CameraPreview";
import { ChatLandingHeadline, ChatLandingSuggestions, useLandingContent } from "./ChatLandingHero";
import { ComposerPlusMenu } from "./ComposerPlusMenu";
import { ComposerVoiceMic } from "./ComposerVoiceMic";
import { ChatUserMessage } from "./ChatUserMessage";
import { ModelActivityPanel } from "./ModelActivityPanel";
import { PresentationQaPanel } from "./PresentationQaPanel";
import { BUILTIN_PRESENTATION_QUERY_COUNT } from "./userPresentationQueries";
import { VisionInspectorPanel } from "./VisionInspectorPanel";
import { SituationSettingsPanel } from "./SituationSettingsPanel";
import { PluginsPanel, type PluginsHubTab } from "./plugins/PluginsPanel";
import { usePluginHealthPoll } from "./plugins/usePluginHealthPoll";
import {
  DEFAULT_VISION_SETTINGS,
  mergeVisionSettings,
  type VisionBehaviorSettings,
} from "./visionSettings";
import {
  ensureVisionLabConfig,
  loadPipelineConfig,
  savePipelineConfig,
} from "./vision-lab/core/configStorage";
import { intervalsFromMode } from "./vision-lab/core/schedule";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";
import { appendModelActivity, type ModelActivityEntry } from "./modelActivityLog";
import { exposeGroveeQaWindow, qaChatBridge } from "./qaChatBridge";
import { SearchProgressPanel } from "./SearchProgressPanel";
import { ContextRing, type ContextUsage } from "./ContextRing";
import { prepareChatContext } from "./chatResourceBudget";
import {
  defaultSystemPromptChars,
  estimateLiveContextUsage,
} from "./contextUsageEstimate";
import {
  detectChatHardwareProfile,
  getProfileBudgets,
  listChatProfiles,
  loadChatProfileOverride,
  saveChatProfileOverride,
  type ChatHardwareProfileId,
} from "./chatHardwareProfile";
import { runWebSearch, needsWebSearch, needsOpenWebEnrichment, wantsCinemaPlotSummaries, warmLiveWorldCache, buildCapabilityLiveReply, buildOpenWebTopicReply, buildWebFallbackNoDataReply, shouldDeliverStructuredLiveReply, isNewsQuery, isCurrencyQuery, isEarthquakeQuery, isDisasterQuery, isAviationQuery, isShipsQuery, isMarineInfraQuery, wantsNewsHeadlineBulletsInChat, clearQueryCache, type SearchIntent, type SearchSourceResult, type SearchBrief, type AnswerShape } from "./webSearch";
import { hasPlaceholderReply } from "./webSearch/openWebQueryPlanner";
import {
  startGroveeNewsBoot,
  useNewsEngineStatus,
} from "./groveeNews/boot";
import { SearchResultsPanel } from "./searchResults/SearchResultsPanel";
import { UiLanguageToggle } from "./ui/UiLanguageToggle";
import { useUiLanguage } from "./ui/useUiLanguage";
import {
  buildPanelSearchPlan,
  buildUnifiedSearchPayload,
  createEmptySearchPayload,
  clearSearchResultsPayload,
  shouldOpenSearchResultsPanel,
  type SearchResultsPayload,
} from "./searchResults";
import {
  buildNewsPanelGuideReply,
  type GroveeNewsCard,
  type NewsSummaryGemmaProgress,
} from "./groveeNews/bridge";
import {
  GEMMA_NEWS_POLISH_SYSTEM,
  GEMMA_SUMMARY_FALLBACK_HE,
  buildGemmaNewsPolishUserPrompt,
  finalizeGemmaNewsSummary,
} from "./groveeNews/gemmaNewsPolish";
import { agentDebugLog } from "./debugAgentLog";
import { isCrossSourceQuery } from "./webSearch/crossSourceIntents";
import { clearSharedRegionCache } from "./webSearch/sharedRegion";
import { isStarlinkRegionalQuery } from "./webSearch/intents";
import { buildWebTopicSearchPlan, planToSearchPlan } from "./webSearch/webTopicQueryPlan";
import {
  buildOpenWebPlannerUserPrompt,
  OPEN_WEB_QUERY_PLANNER_SYSTEM,
  parseOpenWebQueriesJson,
} from "./webSearch/openWebQueryPlanner";
import { getHfToken, setHfToken } from "./webSearch/hf/hfModelSettings";
import { verifyHfToken } from "./webSearch/hf/verifyHfToken";
import { ChatModelPicker } from "./ChatModelPicker";
import {
  readLocalTextReadyIds,
  SMOLLM_HF_MODEL_ID,
  SMOLLM_RACK_ID,
  applyLocalTextDownloadStates,
} from "./modelRack/localTextModels";
import {
  localTextToUiLanguage,
  prepareLocalTextTurnForModel,
} from "./modelRack/localTextTranslate";
import {
  downloadLocalTextModel,
  generateLocalTextChat,
  abortLocalTextGeneration,
  terminateLocalTextWorker,
} from "./modelRack/localTextModelRuntime";
import {
  DEFAULT_LOCAL_TEXT_SETTINGS,
  mergeLocalTextSettings,
  type LocalTextInferenceBackend,
  type LocalTextModelSettings,
} from "./modelRack/localTextModelSettings";
import {
  resolveStartupModelChoice,
  startupChoiceLabelHe,
  type StartupModelChoice,
  type StartupModelPreference,
  type StartupModelRecommendation,
} from "./startupModelProfile";
import {
  buildCapabilitiesOnlyFallbackMessage,
  pickCapabilitiesDefaultRackId,
  type ChatModelAvailability,
} from "./capabilitiesOnlyMode";
import { readTvDeepLink } from "./deepLinks";
import {
  GEMMA_RACK_ID,
  getRackModelById,
  getSelectedModelId,
  loadModelRack,
  pickableRackModels,
  isSelectableInPicker,
  summarizeRackCounts,
  setSelectedModelId as persistSelectedModelId,
  type RackModelEntry,
} from "./modelRack/modelRack";
import { executeRackModel, isLocalTextChatModel, rackModelRunsInChat } from "./modelRack/modelExecution";
import { getChatUiLanguage } from "./ui/useUiLanguage";
import { runTextChatTurnPrelude } from "./chatTurnPrelude";
import {
  buildLocalTextSystemPrompt,
  localTextMaxNewTokens,
} from "./localTextSystemPrompt";
import { refreshCloudModelRack } from "./modelRack/modelRackScan";
import {
  buildSearchPlannerUserPrompt,
  parseSearchPlanJson,
  regexPlanForQuery,
  shouldUseSearchPlanner,
  type SearchPlan,
} from "./webSearch/searchPlanner";
import { registerGlobeLiveSnapshotListener, subscribeLiveWorldSnapshot, pingGlobeForLiveSnapshot, findGlobeIframe } from "./liveWorld";
import { AISSTREAM_KEY_SAVED_EVENT } from "./apiKeys/apiKeyStore";
import { ApiKeysPanelContent } from "./apiKeys/ApiKeysPanel";
import "./apiKeys/apiKeys.css";
import { refreshLivePanelPayload } from "./searchResults/panelSearch";
import {
  fetchStartupContext,
  refreshLocalWeather,
  buildLocalTimeAnswer,
  buildStartupPromptBlock,
  isLocalContextTimeQuery,
  clearStartupContextCache,
  type StartupContext,
} from "./startupContext";
import { LocalContextBar } from "./LocalContextBar";
import { TimeClockWidget } from "./TimeClockWidget";
import {
  buildShortTimeReply,
  buildTimeWidgetFromStartupContext,
  buildTimeWidgetFromWorldTimeSource,
  isSinglePlaceTimeWidgetQuery,
} from "./timeWidget/resolveTimeWidget";
import type { TimeWidgetData } from "./timeWidget/types";
import { GamesPanel } from "./GamesPanel";
import { GlobePanel } from "./GlobePanel";
import { LiveMediaPanel } from "./liveMedia/LiveMediaPanel";
import { buildGlobeCommand, shouldOpenGlobePanel } from "./realityGlobe/intents";
import type { GlobeCommand } from "./realityGlobe/bridge";
import {
  buildGlobeCommandFromSearch,
  shouldOpenGlobeForStructuredGeo,
} from "./realityGlobe/searchGlobeBridge";
import {
  buildGlobePlaceReply,
  buildPlacesMapReply,
  buildRouteMapReply,
  GLOBE_PRESENTATION_APPEND,
} from "./realityGlobe/globePresentation";
import {
  buildGlobeHeadlinePrompt,
  GLOBE_HEADLINE_SYSTEM,
  parseHeadlineLines,
  publishGlobeHeadlineResult,
  subscribeGlobeHeadlineRequests,
} from "./realityGlobe/globeHeadlineBridge";
import { GameCategoryPicker } from "./GameCategoryPicker";
import {
  buildGameSearchFoundReply,
  buildGameSearchNotFoundReply,
  categoryLabelHe,
  GAMES_CATALOG_PAGE_SIZE,
  parseGameUserRequest,
  randomOnlineGames,
  searchOnlineGamesWithFallback,
  shouldOpenGamePanel,
  type GameCategoryId,
  type OnlineGame,
} from "./gameSearch";
import { loadGamesSession, recordGamePlay, saveGamesSession } from "./localExperience/gamesStore";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  modelLabel?: string;
  /** Full HTML/code source — persisted so chat switching does not corrupt preview. */
  artifact?: Artifact;
  /** Native thinking channel — rendered in ThinkingBlock after generation ends. */
  thought?: string;
  /** English camera perception context — shown in collapsible "חושב…" panel, not as chat answer. */
  visionContext?: string;
  /** Thumbnails for user-attached images (session memory; not persisted to localStorage). */
  images?: StoredMessageImage[];
  /** Live web search sources fetched for this turn (Search mode). */
  searchSources?: SearchSourceResult[];
  searchSummary?: string;
  /** Animated clock card for "what time" queries. */
  timeWidget?: TimeWidgetData;
  /** @deprecated games render in side panel only — kept for old saved sessions */
  gameResults?: OnlineGame[];
  /** Show category picker when a specific game search had no match. */
  showGameCategories?: boolean;
  gameBrowseCategory?: GameCategoryId | null;
};

type WorkerOutMessage =
  | { type: "status"; text: string }
  | {
      type: "progress";
      text: string;
      progress: number;
      phase?: "download" | "init";
      loaded?: number;
      total?: number;
      speedBps?: number;
      detail?: string;
      file?: string;
    }
  | { type: "loaded"; modelId: string; device: string }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "aborted" }
  | { type: "error"; error: string; scope?: "chat" | "boot" }
  | {
      type: "scene_analysis";
      requestId: string;
      ok: boolean;
      objects?: string[];
      people?: string[];
      current?: string[];
      events?: string[];
      interesting?: boolean;
      summary?: string;
      raw?: string;
      error?: string;
    }
  | {
      type: "character_utterance";
      requestId: string;
      ok: boolean;
      text?: string;
      error?: string;
    }
  | {
      type: "character_utterance_token";
      requestId: string;
      tokens: number;
      text?: string;
    }
  | {
      type: "search_plan";
      requestId: string;
      ok: boolean;
      text?: string;
      error?: string;
    };

const GEMMA_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const SETTINGS_STORAGE_KEY = "grovee_model_settings_v1";
const WEBGPU_BLOCKED_KEY = "grovee-webgpu-blocked";

const isWebGpuInferenceError = (msg: string) =>
  /WebGPU|bad_alloc|Can't create a session|GatherBlockQuantized|node_embedding_Quant|ERROR_CODE:\s*[69]|Could not find an implementation/i.test(
    msg,
  );
const CHATS_STORAGE_KEY = "grovee_chats_v1";

const qaSearchParam = (): string | null => {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("qa");
};

const isGithubPagesHost = (): boolean => {
  if (typeof window === "undefined") return false;
  return /(?:^|\.)github\.io$/i.test(window.location.hostname);
};

/** QA panel + bridge: local dev, GitHub Pages, or ?qa=1|chat|panel */
const QA_BRIDGE_ENABLED =
  typeof window !== "undefined" &&
  (import.meta.env.DEV ||
    isGithubPagesHost() ||
    ["1", "chat", "panel"].includes(qaSearchParam() ?? ""));

/** Dev-only: ?qa=vision skips Gemma gate for automated face/emotion QA. */
const QA_VISION_MODE =
  import.meta.env.DEV &&
  typeof window !== "undefined" &&
  qaSearchParam() === "vision";

/** ?qa=chat exposes window.__groveeQa for Playwright / console QA. */
void (
  QA_BRIDGE_ENABLED &&
  (qaSearchParam() === "chat" || qaSearchParam() === "1")
);

/** Canned live replies are default; QA panel opt-in enables LLM path per turn only. */
const QA_FORCE_LLM_DEFAULT = false;

/** Friendly product tips while Gemma downloads — explain what GROVEE is. */
const LOADING_DOWNLOAD_TIPS = [
  "GROVEE הוא צ'אט AI חינמי — רץ בדפדפן, בלי התקנה ובלי שליחת שיחות לענן.",
  "שואלים בעברית או באנגלית, מקבלים תשובות ישירות מהדפדפן — המודל Gemma 4 E2B נטען אצלך.",
  "הפרטיות נשארת אצלך: מה שאתה כותב לא עובר לשרת AI חיצוני.",
  "אחרי שהטעינה הראשונה תסתיים, אפשר לשוחח גם בלי חיבור אינטרנט.",
  "אפשר לבקש קוד, הסברים, סיפורים ודפי HTML — GROVEE מייצר הכול מקומית בדפדפן.",
  "זו אפליקציית ווב: המשקולות (~3.9GB, פעם ראשונה) נשמרות במטמון הדפדפן — אין התקנה.",
  "GROVEE בנוי על Transformers.js — AI בדפדפן, כולל ראייה (תמונות) מקומית.",
  "צרף תמונה בכפתור 📎 או הדבק (Ctrl+V) — המודל יתאר ויפענח אותה אצלך במחשב.",
  "כפתור Think מפעיל <|think|> native של Gemma 4; חיפוש ברשת נדלק אוטומטית לשאלות על מזג אוויר, עובדות ומידע עדכני.",
  "הטעינה הראשונה ארוכה רק פעם אחת — בפעם הבאה GROVEE יעלה הרבה יותר מהר מהמטמון.",
  "עוד רגע תוכל לפתוח שיחה חדשה ולדבר עם העוזר המקומי שלך — חינם לגמרי, ישירות מהווב.",
] as const;

const LOADING_INIT_TIPS = [
  "GROVEE כמעט מוכן — טוענים את Gemma לזיכרון (ONNX / WebGPU).",
  "הקבצים כבר אצלך — זה השלב האחרון לפני הצ'אט החינמי.",
  "עוד שנייה ותוכל לשאול כל שאלה — בלי לחכות לענן.",
] as const;

const LOADING_TIP_INTERVAL_MS = 10_000;

type ChatSession = {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
};

type ChatSessionsState = { activeId: string; sessions: ChatSession[] };

type TunableModelSettings = {
  temperature: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  topP: number;
  systemPrompt: string;
};

type InferenceBackendPreference = "auto" | "webgpu" | "wasm";

type AppSettings = {
  hfRemoteHost: string;
  inferenceBackend: InferenceBackendPreference;
  gemma: TunableModelSettings;
  localText: LocalTextModelSettings;
  startupModel: StartupModelPreference;
  vision: VisionBehaviorSettings;
};

const defaultGemmaSettings: TunableModelSettings = {
  temperature: 0.45,
  maxNewTokens: 768,
  repetitionPenalty: 1.12,
  topP: 0.9,
  systemPrompt: GROVEE_CHAT_SYSTEM,
};

/** Auto: WebGPU when the GPU supports Gemma q4, WASM fallback on failure. WASM only when chosen or after a confirmed GPU block. */
const normalizeInferenceBackend = (
  parsed: InferenceBackendPreference | undefined,
): InferenceBackendPreference => {
  if (parsed === "wasm" || parsed === "webgpu" || parsed === "auto") return parsed;
  if (typeof window !== "undefined" && localStorage.getItem(WEBGPU_BLOCKED_KEY) === "1") {
    return "wasm";
  }
  return "auto";
};

const defaultAppSettings = (): AppSettings => ({
  hfRemoteHost: "",
  inferenceBackend: normalizeInferenceBackend(undefined),
  gemma: { ...defaultGemmaSettings },
  localText: { ...DEFAULT_LOCAL_TEXT_SETTINGS },
  startupModel: "auto",
  vision: { ...DEFAULT_VISION_SETTINGS },
});

const loadSettings = (): AppSettings => {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return defaultAppSettings();
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    const mergedGemma = { ...defaultGemmaSettings, ...parsed.gemma };
    const systemPrompt = migrateGemmaSystemPrompt(
      typeof mergedGemma.systemPrompt === "string" ? mergedGemma.systemPrompt : GROVEE_CHAT_SYSTEM,
    );
    const gemma = { ...mergedGemma, systemPrompt };
    let inferenceBackend = normalizeInferenceBackend(parsed.inferenceBackend);
    // Recover Auto after mistaken forced-WASM on GitHub Pages — try WebGPU again on capable GPUs.
    if (
      typeof window !== "undefined" &&
      localStorage.getItem(WEBGPU_BLOCKED_KEY) === "1" &&
      parsed.inferenceBackend === "wasm"
    ) {
      try {
        localStorage.removeItem(WEBGPU_BLOCKED_KEY);
      } catch {
        /* ignore */
      }
      inferenceBackend = "auto";
    }
    const settings: AppSettings = {
      ...defaultAppSettings(),
      hfRemoteHost: typeof parsed.hfRemoteHost === "string" ? parsed.hfRemoteHost : "",
      inferenceBackend,
      gemma,
      localText: mergeLocalTextSettings(parsed.localText),
      startupModel:
        parsed.startupModel === "gemma" ||
        parsed.startupModel === "local-text" ||
        parsed.startupModel === "auto"
          ? parsed.startupModel
          : "auto",
      vision: mergeVisionSettings(parsed.vision),
    };
    const shouldPersist =
      systemPrompt !== mergedGemma.systemPrompt || inferenceBackend !== parsed.inferenceBackend;
    if (shouldPersist) {
      saveSettings(settings);
    }
    return settings;
  } catch {
    return defaultAppSettings();
  }
};

const saveSettings = (s: AppSettings) => {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
};

const sessionTitleFromMessages = (sessionMessages: ChatMessage[]): string => {
  const firstUser = sessionMessages.find((m) => m.role === "user")?.content?.trim();
  if (!firstUser) return "שיחה חדשה";
  return firstUser.slice(0, 28) + (firstUser.length > 28 ? "…" : "");
};

const newChatSessionId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `s-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

const loadChatSessionsState = (): ChatSessionsState => {
  const freshId = newChatSessionId();
  const freshSession: ChatSession = {
    id: freshId,
    title: "שיחה חדשה",
    updatedAt: Date.now(),
    messages: [],
  };

  try {
    const raw = localStorage.getItem(CHATS_STORAGE_KEY);
    if (!raw) return { activeId: freshId, sessions: [freshSession] };
    const parsed = JSON.parse(raw) as { activeId?: string; sessions?: ChatSession[] };
    if (!parsed.sessions || !Array.isArray(parsed.sessions) || parsed.sessions.length === 0) {
      return { activeId: freshId, sessions: [freshSession] };
    }
    const history = parsed.sessions
      .map((s) => ({
        id: typeof s.id === "string" ? s.id : newChatSessionId(),
        title: typeof s.title === "string" ? s.title : "שיחה",
        updatedAt: typeof s.updatedAt === "number" ? s.updatedAt : Date.now(),
        messages: Array.isArray(s.messages) ? s.messages : [],
      }))
      .filter((s) => s.messages.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt);
    return { activeId: freshId, sessions: [freshSession, ...history] };
  } catch {
    return { activeId: freshId, sessions: [freshSession] };
  }
};

const saveChatSessionsState = (state: ChatSessionsState) => {
  const sessions = state.sessions.filter((s) => s.messages.length > 0 || s.id === state.activeId);
  const serializable = { activeId: state.activeId, sessions };
  try {
    localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(serializable));
  } catch {
    try {
      const trimmed = {
        activeId: state.activeId,
        sessions: state.sessions.map((s) => ({
          ...s,
          messages: s.messages.map((m) => ({
            ...m,
            content: m.content.length > 12_000 ? `${m.content.slice(0, 12_000)}…` : m.content,
            artifact:
              m.artifact && m.artifact.content.length > 500_000
                ? { ...m.artifact, content: `${m.artifact.content.slice(0, 500_000)}…` }
                : m.artifact,
          })),
        })),
      };
      localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(trimmed));
    } catch {
      // quota — skip
    }
  }
};

const formatInferenceDevice = (device: string): string => {
  const d = device.toLowerCase();
  if (d === "webgpu") return "WebGPU (GPU)";
  if (d === "wasm") return "WASM (CPU)";
  if (d === "cache") return "cache";
  return device;
};

function ArtifactChip({
  kind,
  label,
  onOpen,
}: {
  kind: "code" | "html";
  label: string;
  onOpen: () => void;
}) {
  return (
    <button type="button" className="artifact-chip" onClick={onOpen}>
      <span className="artifact-chip-icon" aria-hidden="true">
        {kind === "html" ? "◫" : "{}"}
      </span>
      <span className="artifact-chip-text">פתח {label} בחלונית</span>
      <span className="artifact-chip-arrow" aria-hidden="true">
        ←
      </span>
    </button>
  );
}

function CameraTopicBar({ topics }: { topics: string[] }) {
  if (!topics.length) return null;
  const recent = topics.slice(-6);
  return (
    <div className="camera-topic-bar" dir="rtl">
      <span className="camera-topic-bar-label">נושאים:</span>
      {recent.map((t) => (
        <span key={t} className="camera-topic-chip">
          {formatCameraTopicLabel(t)}
        </span>
      ))}
    </div>
  );
}

function VisionContextBlock({ context, streaming }: { context: string; streaming?: boolean }) {
  if (!context.trim()) return null;
  return (
    <details className={`vision-context-block${streaming ? " vision-context-block--live" : ""}`} open={streaming}>
      <summary>{streaming ? "חושב…" : "מה הוא רואה (פנימי)"}</summary>
      <pre className="vision-context-block-body">{context}</pre>
    </details>
  );
}

function ThinkingBlock({ thought, streaming }: { thought: string; streaming?: boolean }) {
  if (!thought.trim()) return null;
  return (
    <details className="thinking-block" open={streaming}>
      <summary>{streaming ? "חושב…" : "תהליך חשיבה"}</summary>
      <pre className="thinking-block-body">{thought}</pre>
    </details>
  );
}

function UserAttachedImages({ images }: { images: StoredMessageImage[] }) {
  if (!images.length) return null;
  return (
    <div className="msg-user-images">
      {images.map((img) => (
        <img key={img.id} className="msg-user-image" src={img.previewUrl} alt="" loading="lazy" />
      ))}
    </div>
  );
}

function MessageBody({
  content,
  onOpenArtifact,
  showThinking = false,
  savedThought,
  savedVisionContext,
  savedArtifact,
  chatOnlyDocument = false,
}: {
  content: string;
  onOpenArtifact?: (artifact: Artifact) => void;
  showThinking?: boolean;
  savedThought?: string;
  savedVisionContext?: string;
  savedArtifact?: Artifact;
  /** Document/image turn — render in chat, no HTML artifact chips. */
  chatOnlyDocument?: boolean;
}) {
  const streamParts = useMemo(() => splitAssistantStream(content, showThinking), [content, showThinking]);
  const thoughtText = showThinking ? streamParts.thought : (savedThought ?? "");
  const displayContent = showThinking ? streamParts.answer : content;
  const parts = useMemo(() => extractRichParts(displayContent), [displayContent]);
  const dir = isRtlText(displayContent || thoughtText) ? "rtl" : "ltr";
  const showVisionCtx = !!savedVisionContext?.trim();
  const visionStreaming = showVisionCtx && showThinking && !displayContent.trim() && !thoughtText.trim();

  return (
    <div className="msg-body" dir={dir}>
      {showVisionCtx ? (
        <VisionContextBlock context={savedVisionContext!} streaming={visionStreaming} />
      ) : null}
      {thoughtText.trim() ? (
        <ThinkingBlock thought={thoughtText} streaming={showThinking && streamParts.thinkingInProgress} />
      ) : null}
      {savedArtifact && onOpenArtifact ? (
        <ArtifactChip
          kind={savedArtifact.kind}
          label={savedArtifact.kind === "html" ? "HTML" : savedArtifact.title}
          onOpen={() => onOpenArtifact(savedArtifact)}
        />
      ) : null}
      {parts.map((part, i) => {
        if (chatOnlyDocument && (part.type === "html" || part.type === "code")) {
          return <ChatMarkdown key={i} text={part.type === "html" ? part.value : `\`\`\`${part.lang ?? ""}\n${part.value}\n\`\`\``} />;
        }
        if (savedArtifact && (part.type === "html" || part.type === "code")) return null;
        if (part.type === "html" && part.value.length > 0) {
          if (onOpenArtifact) {
            return (
              <ArtifactChip
                key={i}
                kind="html"
                label="HTML"
                onOpen={() => onOpenArtifact({ kind: "html", content: part.value, title: "HTML" })}
              />
            );
          }
        }
        if (part.type === "code") {
          if (onOpenArtifact) {
            const lang = part.lang || "code";
            return (
              <ArtifactChip
                key={i}
                kind="code"
                label={lang}
                onOpen={() =>
                  onOpenArtifact({ kind: "code", content: part.value, lang: part.lang, title: lang })
                }
              />
            );
          }
        }
        if (part.type === "image") {
          return (
            <div key={i} className="msg-image-wrap">
              <img className="msg-image" src={part.value} alt="Generated" loading="lazy" />
            </div>
          );
        }
        return (
          <ChatMarkdown key={i} text={part.value} />
        );
      })}
    </div>
  );
}

type SettingsModalTab = "gemma" | "localText" | "vision" | "api-keys";

function SettingsModal({
  open,
  onClose,
  settings,
  onSave,
  onClearCache,
  cacheClearing,
  initialTab = "gemma",
}: {
  open: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSave: (s: AppSettings) => void;
  onClearCache: () => void;
  cacheClearing: boolean;
  initialTab?: SettingsModalTab;
}) {
  const [draft, setDraft] = useState<AppSettings>(() => settings);
  const [settingsTab, setSettingsTab] = useState<SettingsModalTab>(initialTab);
  const [hfTokenDraft, setHfTokenDraft] = useState(() => getHfToken() ?? "");
  const [hfTokenStatus, setHfTokenStatus] = useState<string | null>(null);
  const [hfTokenChecking, setHfTokenChecking] = useState(false);
  const [chatProfile, setChatProfile] = useState<ChatHardwareProfileId>(
    () => loadChatProfileOverride() ?? detectChatHardwareProfile(),
  );

  useEffect(() => {
    if (open) setSettingsTab(initialTab);
  }, [open, initialTab]);

  if (!open) return null;

  const setBackend = (inferenceBackend: InferenceBackendPreference) => {
    setDraft((d) => ({ ...d, inferenceBackend }));
  };

  const patchVision = (partial: Partial<VisionBehaviorSettings>) => {
    setDraft((d) => ({ ...d, vision: { ...d.vision, ...partial } }));
  };

  const setLocalTextBackend = (inferenceBackend: LocalTextInferenceBackend) => {
    setDraft((d) => ({ ...d, localText: { ...d.localText, inferenceBackend } }));
  };

  const settingsHeadline =
    settingsTab === "api-keys"
      ? { title: "מפתחות API", sub: "TMDB · AIS חי · Tavily · Scavio" }
      : settingsTab === "localText"
      ? { title: "הגדרות SmolLM", sub: "SmolLM2 360M · מקומי בדפדפן" }
      : settingsTab === "vision"
        ? { title: "הגדרות מצלמה", sub: "עיניים ואסיטואציות" }
        : { title: "הגדרות Gemma", sub: "GEMMA 4 E2B · מקומי בדפדפן" };

  const settingsBadge =
    settingsTab === "api-keys"
      ? "🔑"
      : settingsTab === "localText"
        ? "S"
        : settingsTab === "vision"
          ? "👁"
          : "G";

  const localTextBackendOptions: { id: LocalTextInferenceBackend; label: string; hint: string }[] = [
    { id: "auto", label: "Auto", hint: "WebGPU אם אפשר; נופל ל-WASM בשגיאה" },
    { id: "wasm", label: "WASM", hint: "מעבד — יציב לשיחות ארוכות" },
    { id: "webgpu", label: "WebGPU", hint: "GPU — מהיר; עלול להיכשל אחרי זמן" },
  ];

  const backendOptions: { id: InferenceBackendPreference; label: string; hint: string }[] = [
    { id: "auto", label: "Auto", hint: "WebGPU אם אפשר; נופל ל-WASM בשגיאה" },
    { id: "wasm", label: "WASM", hint: "מעבד — יציב לשיחות ארוכות" },
    { id: "webgpu", label: "WebGPU", hint: "GPU — מהיר; עלול להיכשל אחרי זמן" },
  ];

  return (
    <div
      className="settings-overlay modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="settings-panel modal-box">
        <div className="settings-head">
          <div className="settings-head-brand">
            <span className="settings-head-badge">{settingsBadge}</span>
            <div>
              <h2 id="settings-title">{settingsHeadline.title}</h2>
              <p className="settings-head-sub">{settingsHeadline.sub}</p>
            </div>
          </div>
          <button type="button" className="icon-close settings-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </div>

        <div className="settings-tabs" role="tablist" aria-label="קטגוריות הגדרות">
          <button
            type="button"
            role="tab"
            aria-selected={settingsTab === "gemma"}
            className={`settings-tab ${settingsTab === "gemma" ? "active" : ""}`}
            onClick={() => setSettingsTab("gemma")}
          >
            Gemma
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={settingsTab === "localText"}
            className={`settings-tab ${settingsTab === "localText" ? "active" : ""}`}
            onClick={() => setSettingsTab("localText")}
          >
            SmolLM
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={settingsTab === "vision"}
            className={`settings-tab ${settingsTab === "vision" ? "active" : ""}`}
            onClick={() => setSettingsTab("vision")}
          >
            עיניים ואסיטואציות
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={settingsTab === "api-keys"}
            className={`settings-tab ${settingsTab === "api-keys" ? "active" : ""}`}
            onClick={() => setSettingsTab("api-keys")}
          >
            מפתחות API
          </button>
        </div>

        {settingsTab === "api-keys" ? (
          <div className="settings-api-keys">
            <ApiKeysPanelContent active showSoon={false} />
          </div>
        ) : null}

        {settingsTab === "vision" ? (
          <SituationSettingsPanel vision={draft.vision} onVisionChange={patchVision} />
        ) : null}

        {settingsTab === "localText" ? (
        <>
        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            מנוע חישוב
          </h3>
          <div className="settings-backend-pills" role="radiogroup" aria-label="מנוע חישוב SmolLM">
            {localTextBackendOptions.map((opt) => (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={draft.localText.inferenceBackend === opt.id}
                className={`settings-backend-pill ${draft.localText.inferenceBackend === opt.id ? "active" : ""}`}
                onClick={() => setLocalTextBackend(opt.id)}
              >
                <span className="settings-backend-pill-label">{opt.label}</span>
                <span className="settings-backend-pill-hint">{opt.hint}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            פרמטרי המודל
          </h3>
          <div className="settings-grid">
            <label className="settings-field">
              <span className="settings-field-label">טמפרטורה</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={2}
                value={draft.localText.temperature}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, temperature: Number(e.target.value) },
                  }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">מקסימום טוקנים (שיחה)</span>
              <input
                type="number"
                min={32}
                max={1024}
                value={draft.localText.maxNewTokens}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, maxNewTokens: Number(e.target.value) },
                  }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">מקסימום טוקנים (חיפוש)</span>
              <input
                type="number"
                min={64}
                max={1024}
                value={draft.localText.maxNewTokensSearch}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, maxNewTokensSearch: Number(e.target.value) },
                  }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">Top P</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={draft.localText.topP}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, topP: Number(e.target.value) },
                  }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">טוקנים (ברכה קצרה)</span>
              <input
                type="number"
                min={16}
                max={256}
                value={draft.localText.maxNewTokensGreeting}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, maxNewTokensGreeting: Number(e.target.value) },
                  }))
                }
              />
            </label>
          </div>
          <label className="settings-field settings-field--full">
            <span className="settings-field-label">הנחיה ראשית (System prompt)</span>
            <textarea
              rows={5}
              className="settings-prompt-area"
              value={draft.localText.systemPrompt}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  localText: { ...d.localText, systemPrompt: e.target.value },
                }))
              }
              dir="ltr"
            />
          </label>
        </section>

        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            אורך שיחה והקשר
          </h3>
          <p className="settings-danger-note" style={{ marginBottom: 12 }}>
            SmolLM קטן — הקטנת היסטוריה והקשר חיפוש משפרת יציבות. ממשק עברית מתורגם לאנגלית לפני המודל.
          </p>
          <div className="settings-grid">
            <label className="settings-field">
              <span className="settings-field-label">תורות היסטוריה</span>
              <input
                type="number"
                min={2}
                max={24}
                value={draft.localText.historyTurns}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, historyTurns: Number(e.target.value) },
                  }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">תווים מהחיפוש (web brief)</span>
              <input
                type="number"
                min={200}
                max={2000}
                step={50}
                value={draft.localText.webBriefChars}
                onChange={(e) =>
                  setDraft((d) => ({
                    ...d,
                    localText: { ...d.localText, webBriefChars: Number(e.target.value) },
                  }))
                }
              />
            </label>
          </div>
        </section>

        <section className="settings-card settings-card--danger">
          <h3 className="settings-card-title">
            <span className="settings-card-dot settings-card-dot--warn" aria-hidden="true" />
            איפוס worker
          </h3>
          <p className="settings-danger-note">
            מנתק את SmolLM מהזיכרון — ייטען מחדש לפי מנוע החישוב בהרצה הבאה. לא מוחק את הקבצים שהורדו (
            {SMOLLM_HF_MODEL_ID}).
          </p>
          <button
            type="button"
            className="settings-btn-danger"
            onClick={() => terminateLocalTextWorker()}
          >
            איפוס worker SmolLM
          </button>
        </section>
        </>
        ) : null}

        {settingsTab === "gemma" ? (
        <>
        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            מודל בפתיחה
          </h3>
          <p className="settings-danger-note" style={{ marginBottom: 12 }}>
            אוטומטי: מובייל / זיכרון נמוך / ללא WebGPU → SmolLM (~220MB). מחשב חזק → Gemma (~3.9GB).
          </p>
          <div className="settings-backend-pills" role="radiogroup" aria-label="מודל בפתיחה">
            {(
              [
                { id: "auto" as const, label: "אוטומטי", hint: "לפי זיהוי מכשיר" },
                { id: "gemma" as const, label: "Gemma 4 E2B", hint: "תמיד המודל הגדול" },
                { id: "local-text" as const, label: "SmolLM2", hint: "תמיד המודל הקטן" },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={draft.startupModel === opt.id}
                className={`settings-backend-pill ${draft.startupModel === opt.id ? "active" : ""}`}
                onClick={() => setDraft((d) => ({ ...d, startupModel: opt.id }))}
              >
                <span className="settings-backend-pill-label">{opt.label}</span>
                <span className="settings-backend-pill-hint">{opt.hint}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            מנוע חישוב
          </h3>
          <div className="settings-backend-pills" role="radiogroup" aria-label="מנוע חישוב">
            {backendOptions.map((opt) => (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={draft.inferenceBackend === opt.id}
                className={`settings-backend-pill ${draft.inferenceBackend === opt.id ? "active" : ""}`}
                onClick={() => setBackend(opt.id)}
              >
                <span className="settings-backend-pill-label">{opt.label}</span>
                <span className="settings-backend-pill-hint">{opt.hint}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            פרמטרי המודל
          </h3>
          <div className="settings-grid">
            <label className="settings-field">
              <span className="settings-field-label">טמפרטורה</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={2}
                value={draft.gemma.temperature}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, gemma: { ...d.gemma, temperature: Number(e.target.value) } }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">מקסימום טוקנים</span>
              <input
                type="number"
              min={32}
              max={4096}
                value={draft.gemma.maxNewTokens}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, gemma: { ...d.gemma, maxNewTokens: Number(e.target.value) } }))
                }
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">Top P</span>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={draft.gemma.topP}
                onChange={(e) => setDraft((d) => ({ ...d, gemma: { ...d.gemma, topP: Number(e.target.value) } }))}
              />
            </label>
            <label className="settings-field">
              <span className="settings-field-label">קנס חזרות</span>
              <input
                type="number"
                step={0.02}
                min={1}
                max={2}
                value={draft.gemma.repetitionPenalty}
                onChange={(e) =>
                  setDraft((d) => ({ ...d, gemma: { ...d.gemma, repetitionPenalty: Number(e.target.value) } }))
                }
              />
            </label>
          </div>
          <label className="settings-field settings-field--full">
            <span className="settings-field-label">הנחיה ראשית (System prompt)</span>
            <textarea
              rows={5}
              className="settings-prompt-area"
              value={draft.gemma.systemPrompt}
              onChange={(e) => setDraft((d) => ({ ...d, gemma: { ...d.gemma, systemPrompt: e.target.value } }))}
              dir="ltr"
            />
          </label>
        </section>

        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            פרופיל אוויר בשיחה
          </h3>
          <p className="settings-danger-note" style={{ marginBottom: 12 }}>
            תקציב prompt ו-max tokens לחיפוש — לפי RAM (Chrome מדווח buckets).
          </p>
          <div className="settings-backend-pills" role="radiogroup" aria-label="פרופיל חומרה">
            {listChatProfiles().map((p) => (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={chatProfile === p.id}
                className={`settings-backend-pill ${chatProfile === p.id ? "active" : ""}`}
                onClick={() => setChatProfile(p.id)}
              >
                <span className="settings-backend-pill-label">{p.labelHe}</span>
                <span className="settings-backend-pill-hint">
                  ~{(p.totalPromptChars / 1000).toFixed(0)}k chars · search {p.maxNewTokensSearch} tok
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="settings-card">
          <h3 className="settings-card-title">
            <span className="settings-card-dot" aria-hidden="true" />
            Hugging Face API
          </h3>
          <p className="settings-danger-note" style={{ marginBottom: 12 }}>
            אופציונלי — לחיפוש HF ולהרצת מודלים דרך Inference API (לא לסריקת הרשימה בלי מפתח).
            נשמר מקומית בדפדפן.
          </p>
          <label className="settings-field settings-field--full">
            <span className="settings-field-label">HF Token</span>
            <input
              type="password"
              autoComplete="off"
              placeholder="hf_…"
              value={hfTokenDraft}
              onChange={(e) => {
                setHfTokenDraft(e.target.value);
                setHfTokenStatus(null);
              }}
              dir="ltr"
            />
          </label>
          <div className="settings-hf-token-actions">
            <button
              type="button"
              className="settings-btn-ghost"
              disabled={hfTokenChecking}
              onClick={() => {
                setHfToken(hfTokenDraft.trim());
                setHfTokenStatus(
                  hfTokenDraft.trim() ? "✓ Token נשמר מקומית" : "Token נמחק מהדפדפן",
                );
              }}
            >
              שמור token
            </button>
            <button
              type="button"
              className="settings-btn-save settings-btn-save--compact"
              disabled={hfTokenChecking || !hfTokenDraft.trim()}
              onClick={() => {
                void (async () => {
                  setHfTokenChecking(true);
                  setHfTokenStatus("בודק מול Hugging Face Hub…");
                  const result = await verifyHfToken(hfTokenDraft);
                  if (result.ok) {
                    setHfToken(hfTokenDraft.trim());
                    setHfTokenStatus(`✓ חיבור תקין — שלום ${result.username}`);
                  } else {
                    setHfTokenStatus(`✗ ${result.message}`);
                  }
                  setHfTokenChecking(false);
                })();
              }}
            >
              {hfTokenChecking ? "בודק…" : "בדוק חיבור"}
            </button>
          </div>
          {hfTokenStatus ? (
            <p
              className={`settings-hf-token-status${hfTokenStatus.startsWith("✓") ? " is-ok" : hfTokenStatus.startsWith("✗") ? " is-err" : ""}`}
              dir="auto"
            >
              {hfTokenStatus}
            </p>
          ) : null}
        </section>

        <section className="settings-card settings-card--danger">
          <h3 className="settings-card-title">
            <span className="settings-card-dot settings-card-dot--warn" aria-hidden="true" />
            מטמון המודל
          </h3>
          <p className="settings-danger-note">מוחק ~3.9GB של משקולות Gemma (טקסט + ראייה) מהדפדפן. יידרש להוריד מחדש.</p>
          <button type="button" className="settings-btn-danger" onClick={onClearCache} disabled={cacheClearing}>
            {cacheClearing ? "מנקה מטמון…" : "נקה מטמון מודל"}
          </button>
        </section>
        </>
        ) : null}

        <div className="settings-footer">
          {settingsTab === "api-keys" ? (
            <button type="button" className="settings-btn-save" onClick={onClose}>
              סגור
            </button>
          ) : (
            <>
              <button type="button" className="settings-btn-ghost" onClick={() => setDraft(defaultAppSettings())}>
                איפוס ברירת מחדל
              </button>
              <button
                type="button"
                className="settings-btn-save"
                onClick={() => {
                  saveChatProfileOverride(chatProfile);
                  setHfToken(hfTokenDraft);
                  onSave(draft);
                  onClose();
                }}
              >
                שמור
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  const workerRef = useRef<Worker | null>(null);
  const assistantBufferRef = useRef("");
  const pendingVisionContextRef = useRef("");
  const streamTokenCountRef = useRef(0);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageBytesCacheRef = useRef<Map<string, { bytes: ArrayBuffer; mime: string }>>(new Map());
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<CameraStreamHandle | null>(null);
  const cameraLoopRef = useRef<GroveeVisionRunner | null>(null);
  const cameraBootingRef = useRef(false);
  const cameraVisionBudgetRef = useRef(detectVisionBudget());
  const worldMemoryRef = useRef(new WorldMemory());
  const characterBrainRef = useRef(new CharacterBrain());
  const chatTopicRef = useRef<ChatTopic | null>(null);
  const sceneAnalysisResolversRef = useRef(
    new Map<string, (result: SceneAnalysisResult | null) => void>(),
  );
  const characterUtteranceResolversRef = useRef(new Map<string, (text: string | null) => void>());
  const characterUtteranceTokenListenersRef = useRef(
    new Map<string, { onCount?: (tokens: number) => void; onChunk?: (text: string) => void }>(),
  );
  const searchPlanResolversRef = useRef(new Map<string, (raw: string | null) => void>());
  const workerInferenceBusyRef = useRef(false);
  const globeHeadlineModeRef = useRef(false);
  const globeHeadlineBufferRef = useRef("");
  const sceneAnalysisMetaRef = useRef(new Map<string, { reason?: string }>());

  const [appSettings, setAppSettings] = useState<AppSettings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<SettingsModalTab>("gemma");
  const [settingsModalKey, setSettingsModalKey] = useState(0);
  const [modelRack, setModelRack] = useState<RackModelEntry[]>(() => loadModelRack());
  const [selectedRackModelId, setSelectedRackModelId] = useState(() => getSelectedModelId());
  const [localTextDownloadingId, setLocalTextDownloadingId] = useState<string | null>(null);
  const [localTextDownloadPct, setLocalTextDownloadPct] = useState(0);
  const [localTextDownloadLabel, setLocalTextDownloadLabel] = useState("");
  const modelRackRef = useRef(modelRack);
  modelRackRef.current = modelRack;
  const selectedRackModelRef = useRef(selectedRackModelId);
  selectedRackModelRef.current = selectedRackModelId;
  const [pluginsOpen, setPluginsOpen] = useState(false);
  const [pluginsHubTab, setPluginsHubTab] = useState<PluginsHubTab>("plugins");
  const pluginHealth = usePluginHealthPoll(true);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isGemmaLoaded, setIsGemmaLoaded] = useState(false);
  const [bootTarget, setBootTarget] = useState<StartupModelChoice>("gemma");
  const [chatModelAvailability, setChatModelAvailability] =
    useState<ChatModelAvailability>("gemma");
  const [capabilitiesFailureReason, setCapabilitiesFailureReason] = useState<string | null>(
    null,
  );
  const [startupRecommendation, setStartupRecommendation] =
    useState<StartupModelRecommendation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const focusComposerInput = useCallback(() => {
    queueMicrotask(() => {
      const el = textareaRef.current;
      if (!el || el.disabled) return;
      el.focus({ preventScroll: true });
    });
  }, []);
  const [status, setStatus] = useState("Not loaded");
  const [progress, setProgress] = useState(0);
  const [loadingPhase, setLoadingPhase] = useState<"download" | "init">("download");
  const [loadingBytes, setLoadingBytes] = useState({ loaded: 0, total: 0, speedBps: 0 });
  const [loadingFile, setLoadingFile] = useState("");
  const [loadingTipIndex, setLoadingTipIndex] = useState(0);
  const [workerBootError, setWorkerBootError] = useState<string | null>(null);
  const [workerReloadKey, setWorkerReloadKey] = useState(0);
  const [cacheClearing, setCacheClearing] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<PendingAttachment[]>([]);
  const [attachProcessing, setAttachProcessing] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [chatSessionsState, setChatSessionsState] = useState<ChatSessionsState>(() => loadChatSessionsState());
  const [cameraStore, setCameraStore] = useState<CameraSessionStore>(() => loadCameraSessionStore());
  const [cameraHistorySearch, setCameraHistorySearch] = useState("");
  const cameraStoreRef = useRef(cameraStore);
  const generationCameraModeRef = useRef(false);
  const generationChatOnlyDocumentRef = useRef(false);
  const [chatOnlyDocumentMode, setChatOnlyDocumentMode] = useState(false);
  const lastTurnUsedVisionRef = useRef(false);
  const [assistantBuffer, setAssistantBuffer] = useState("");
  const [streamingVisionContext, setStreamingVisionContext] = useState("");
  const [streamTokenCount, setStreamTokenCount] = useState(0);
  const [thinkingMode, setThinkingMode] = useState(false);
  const [streamingSearchSources, setStreamingSearchSources] = useState<{
    sources: SearchSourceResult[];
    summary: string;
    query?: string;
    brief?: SearchBrief;
    active?: boolean;
  } | null>(null);
  const [contextRefreshKey, setContextRefreshKey] = useState(0);
  const measuredSystemPromptCharsRef = useRef(0);
  const measuredWebContextCharsRef = useRef(0);
  const [gamesPanelOpen, setGamesPanelOpen] = useState(false);
  const [globePanelOpen, setGlobePanelOpen] = useState(false);
  const [liveMediaPanelOpen, setLiveMediaPanelOpen] = useState(() => readTvDeepLink());
  const [globeCommand, setGlobeCommand] = useState<GlobeCommand | null>(null);
  const [gamesPanelGames, setGamesPanelGames] = useState<OnlineGame[]>([]);
  const [gamesPanelTitle, setGamesPanelTitle] = useState("משחקים און־ליין");
  const [gamesPanelLoading, setGamesPanelLoading] = useState(false);
  const [gamesEmbedGame, setGamesEmbedGame] = useState<OnlineGame | null>(null);
  const [gamesPanelCategory, setGamesPanelCategory] = useState<GameCategoryId>("featured");
  const [gamesPanelStartView, setGamesPanelStartView] = useState<"browse" | "recent" | "favorites">("browse");
  const [gamesPanelLayout, setGamesPanelLayout] = useState<"side" | "full">("side");
  const [streamingGameCategoryPicker, setStreamingGameCategoryPicker] = useState(false);
  const [cameraMode, setCameraMode] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraObserving, setCameraObserving] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("");
  const [characterMood, setCharacterMood] = useState<CharacterMood>("observing");
  const [halMoodState, setHalMoodState] = useState<HalMoodState | null>(null);
  const [halInterpretation, setHalInterpretation] = useState<import("./vision2/types").InterpretationLayer | null>(null);
  const [halConsciousness, setHalConsciousness] = useState<import("./vision2/types").ConsciousnessLayer | null>(null);
  const [halEntity, setHalEntity] = useState<import("./vision2/entityProfile").EntityProfile | null>(null);
  const [activityLogOpen, setActivityLogOpen] = useState(false);
  const [presentationQaOpen, setPresentationQaOpen] = useState(
    () =>
      QA_BRIDGE_ENABLED &&
      qaSearchParam() === "panel",
  );
  const [visionInspectorOpen, setVisionInspectorOpen] = useState(false);
  const [visionPipelineProgress, setVisionPipelineProgress] = useState("");
  const [visionResult, setVisionResult] = useState<VisionResult>(() => ({
    objects: [],
    poseLandmarks: [],
    poseActions: [],
    hands: [],
    fingerStates: [],
    staticGestures: [],
    motionGestures: [],
    faces: [],
    emotion: null,
    interactions: [],
    events: [],
    bodyLanguage: [],
    environment: "Unknown",
    sceneDescription: "Start the camera to begin analysis.",
    vlmDescription: "",
    fps: 0,
    backend: "wasm",
    faceModule: {
      status: "idle",
      message: "Face module not started",
      lastScanAt: 0,
      lastFaceCount: 0,
      modelSource: "none",
    },
  }));
  const [visionPipelineConfig, setVisionPipelineConfig] = useState<PipelineConfig>(() =>
    loadPipelineConfig(),
  );
  const [worldMemorySnapshot, setWorldMemorySnapshot] = useState(() =>
    worldMemoryRef.current.toInspectorSnapshot(),
  );
  const [activityLog, setActivityLog] = useState<ModelActivityEntry[]>([]);
  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [startupContext, setStartupContext] = useState<StartupContext | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [artifactOpen, setArtifactOpen] = useState(false);
  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null);
  const [searchResultsOpen, setSearchResultsOpen] = useState(false);
  const [searchResultsPayload, setSearchResultsPayload] = useState<SearchResultsPayload | null>(null);
  const [searchPanelLoading, setSearchPanelLoading] = useState(false);
  const [desktopLayout, setDesktopLayout] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 769px)").matches,
  );

  const uiLang = useUiLanguage();

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 769px)");
    const sync = () => setDesktopLayout(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    const off = registerGlobeLiveSnapshotListener();
    void warmLiveWorldCache();
    void startGroveeNewsBoot();
    const warmId = window.setInterval(() => void warmLiveWorldCache(), 90_000);
    const onAisKeySaved = () => {
      void warmLiveWorldCache();
      pingGlobeForLiveSnapshot();
      findGlobeIframe()?.contentWindow?.postMessage({ source: "grovee", type: "refreshLiveData" }, "*");
      setSearchResultsPayload((prev) => (prev ? refreshLivePanelPayload(prev) : prev));
    };
    window.addEventListener(AISSTREAM_KEY_SAVED_EVENT, onAisKeySaved);
    return () => {
      off();
      window.clearInterval(warmId);
      window.removeEventListener(AISSTREAM_KEY_SAVED_EVENT, onAisKeySaved);
    };
  }, []);

  useEffect(() => {
    return subscribeLiveWorldSnapshot(() => {
      setSearchResultsPayload((prev) => (prev ? refreshLivePanelPayload(prev) : prev));
    });
  }, []);

  const { status: newsEngineStatus } = useNewsEngineStatus();

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const ctx = await fetchStartupContext();
        if (cancelled) return;
        setStartupContext(ctx);
        const withWx = await refreshLocalWeather(ctx);
        if (!cancelled) setStartupContext(withWx);
      } catch {
        /* optional — UI works without context */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const appSettingsRef = useRef(appSettings);
  const thinkingRef = useRef(thinkingMode);
  const pendingWebSearchRef = useRef<{
    sources: SearchSourceResult[];
    summary: string;
    query?: string;
    answerShape?: AnswerShape;
    crossSource?: boolean;
  } | null>(null);
  const pendingTimeWidgetRef = useRef<TimeWidgetData | null>(null);
  const pendingGameCategoryPickerRef = useRef(false);
  const pendingGameBrowseCategoryRef = useRef<GameCategoryId | null>(null);
  const cameraModeRef = useRef(cameraMode);
  const visionResultRef = useRef(visionResult);
  const isLoadingRef = useRef(isLoading);
  const isGeneratingRef = useRef(isGenerating);
  const isLoadedRef = useRef(isLoaded);
  const isGemmaLoadedRef = useRef(isGemmaLoaded);
  const chatModelAvailabilityRef = useRef(chatModelAvailability);
  const activityLogRef = useRef<ModelActivityEntry[]>([]);
  const qaForceLlmRef = useRef(QA_FORCE_LLM_DEFAULT);
  const qaTurnForceLlmRef = useRef(false);
  const qaEmptyNextSendRef = useRef(false);
  const lastChatSignalRef = useRef(0);
  const messagesListRef = useRef<HTMLDivElement | null>(null);
  const continueModeRef = useRef(false);
  const loadingFileRef = useRef("");
  const wasmBootRetryRef = useRef(false);

  useEffect(() => {
    appSettingsRef.current = appSettings;
  }, [appSettings]);
  useEffect(() => {
    thinkingRef.current = thinkingMode;
  }, [thinkingMode]);
  useEffect(() => {
    cameraModeRef.current = cameraMode;
  }, [cameraMode]);
  useEffect(() => {
    visionResultRef.current = visionResult;
  }, [visionResult]);
  useEffect(() => {
    isLoadingRef.current = isLoading;
  }, [isLoading]);
  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);
  useEffect(() => {
    isLoadedRef.current = isLoaded;
  }, [isLoaded]);

  useEffect(() => {
    isGemmaLoadedRef.current = isGemmaLoaded;
  }, [isGemmaLoaded]);

  useEffect(() => {
    chatModelAvailabilityRef.current = chatModelAvailability;
  }, [chatModelAvailability]);
  useEffect(() => {
    activityLogRef.current = activityLog;
  }, [activityLog]);

  useEffect(() => {
    cameraStoreRef.current = cameraStore;
  }, [cameraStore]);

  useEffect(() => {
    characterBrainRef.current.importSnapshot(cameraStore.memory);
  }, []);

  const persistCameraMemory = useCallback(() => {
    const snap = characterBrainRef.current.exportSnapshot();
    setCameraStore((prev) => {
      const next: CameraSessionStore = {
        ...prev,
        memory: { ...snap, topicLog: prev.memory.topicLog },
        updatedAt: Date.now(),
      };
      saveCameraSessionStore(next);
      return next;
    });
  }, []);

  const setCameraMessages = useCallback(
    (updater: CameraMessage[] | ((prev: CameraMessage[]) => CameraMessage[])) => {
      setCameraStore((prev) => {
        const nextMessages =
          typeof updater === "function"
            ? (updater as (p: CameraMessage[]) => CameraMessage[])(prev.messages)
            : updater;
        const next: CameraSessionStore = {
          ...prev,
          messages: nextMessages,
          updatedAt: Date.now(),
        };
        saveCameraSessionStore(next);
        return next;
      });
    },
    [],
  );

  const appendCameraAssistantMessage = useCallback(
    (params: {
      content: string;
      kind: CameraMessage["kind"];
      modelLabel?: string;
      thought?: string;
      visionContext?: string;
    }) => {
      setCameraMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: params.kind,
          content: params.content,
          ts: Date.now(),
          modelLabel: params.modelLabel ?? "HAL",
          thought: params.thought,
          visionContext: params.visionContext,
        },
      ]);
    },
    [setCameraMessages],
  );

  useEffect(() => {
    saveChatSessionsState(chatSessionsState);
  }, [chatSessionsState]);

  const activeSession = useMemo(
    () =>
      chatSessionsState.sessions.find((s) => s.id === chatSessionsState.activeId) ?? chatSessionsState.sessions[0],
    [chatSessionsState],
  );
  const messages = activeSession.messages;
  const cameraMessages = cameraStore.messages;
  const displayMessages = cameraMode ? cameraMessages : messages;

  const cameraSearchHits = useMemo(
    () =>
      cameraHistorySearch.trim().length >= 2
        ? searchCameraHistory(cameraMessages, cameraHistorySearch, 8)
        : [],
    [cameraMessages, cameraHistorySearch],
  );

  const handleSaveCameraProfile = useCallback((patch: Partial<UserProfile>) => {
    setCameraStore((prev) => updateUserProfile(prev, patch));
  }, []);

  const sortedSessions = useMemo(
    () => [...chatSessionsState.sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [chatSessionsState.sessions],
  );
  const visibleTextSessions = useMemo(
    () => sortedSessions.filter((s) => s.messages.length > 0),
    [sortedSessions],
  );

  const deleteChatSession = useCallback(
    (sessionId: string) => {
      if (isGenerating) return;
      setChatSessionsState((st) => {
        const remaining = st.sessions.filter((s) => s.id !== sessionId);
        if (remaining.length === 0) {
          const id = newChatSessionId();
          return {
            activeId: id,
            sessions: [{ id, title: "שיחה חדשה", updatedAt: Date.now(), messages: [] }],
          };
        }
        const switching = st.activeId === sessionId;
        if (switching) {
          setAssistantBuffer("");
          assistantBufferRef.current = "";
          setArtifactOpen(false);
        }
        return {
          activeId: switching ? remaining[0].id : st.activeId,
          sessions: remaining,
        };
      });
    },
    [isGenerating],
  );

  const handleNewChat = useCallback(() => {
    if (cameraMode) {
      if (isGenerating) return;
      const fresh = clearCameraSessionStore();
      setCameraStore(fresh);
      characterBrainRef.current.reset();
      setSidebarOpen(false);
      setLiveMediaPanelOpen(false);
      return;
    }
    const id = newChatSessionId();
    setPendingAttachments((prev) => {
      for (const p of prev) revokePendingAttachment(p);
      return [];
    });
    setAttachError(null);
    setChatSessionsState((s) => ({
      activeId: id,
      sessions: [{ id, title: "שיחה חדשה", updatedAt: Date.now(), messages: [] }, ...s.sessions],
    }));
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setPrompt("");
    setEditingMessageId(null);
    setEditDraft("");
    setSidebarOpen(false);
    setArtifactOpen(false);
    setLiveMediaPanelOpen(false);
  }, [cameraMode, isGenerating]);

  const setMessages = useCallback((updater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
    setChatSessionsState((st) => {
      const sessions = st.sessions.map((s) => {
        if (s.id !== st.activeId) return s;
        const next =
          typeof updater === "function" ? (updater as (p: ChatMessage[]) => ChatMessage[])(s.messages) : updater;
        return {
          ...s,
          messages: next,
          updatedAt: Date.now(),
          title: sessionTitleFromMessages(next),
        };
      });
      return { ...st, sessions };
    });
  }, []);

  const phase = isLoaded ? "ready" : isLoading ? "loading" : "start";

  useEffect(() => {
    if (phase !== "start") return;
    let cancelled = false;
    void (async () => {
      const rec = await resolveStartupModelChoice(appSettingsRef.current.startupModel);
      if (cancelled) return;
      setStartupRecommendation(rec);
      setBootTarget(rec.choice);
    })();
    return () => {
      cancelled = true;
    };
  }, [phase]);
  const showLanding = phase === "ready" && displayMessages.length === 0 && !assistantBuffer;
  const landingContent = useLandingContent();
  const loadingByteLine = useMemo(() => {
    if (loadingPhase === "init" || loadingBytes.total <= 0) return "";
    const speed =
      loadingBytes.speedBps > 0 ? ` · ~${formatBytes(loadingBytes.speedBps)}/s` : "";
    return `${formatBytes(loadingBytes.loaded)} / ${formatBytes(loadingBytes.total)}${speed}`;
  }, [loadingPhase, loadingBytes]);

  const activeLoadingTips = loadingPhase === "init" ? LOADING_INIT_TIPS : LOADING_DOWNLOAD_TIPS;
  const loadingTip = activeLoadingTips[loadingTipIndex % activeLoadingTips.length];

  useEffect(() => {
    if (phase !== "loading" || !isLoading) {
      queueMicrotask(() => setLoadingTipIndex(0));
      return;
    }
    setLoadingTipIndex(0);
    const tips = loadingPhase === "init" ? LOADING_INIT_TIPS : LOADING_DOWNLOAD_TIPS;
    const id = window.setInterval(() => {
      setLoadingTipIndex((i) => (i + 1) % tips.length);
    }, LOADING_TIP_INTERVAL_MS);
    return () => clearInterval(id);
  }, [phase, isLoading, loadingPhase]);

  useEffect(() => {
    const el = messagesListRef.current;
    if (!el || phase !== "ready") return;
    el.scrollTop = el.scrollHeight;
  }, [displayMessages, assistantBuffer, phase]);

  useEffect(() => {
    if (!isGenerating && isLoaded) focusComposerInput();
  }, [isGenerating, isLoaded, focusComposerInput]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el || phase !== "ready") return;
    const minH = showLanding ? 32 : 36;
    el.style.height = "auto";
    el.style.height = `${Math.min(Math.max(el.scrollHeight, minH), 120)}px`;
  }, [prompt, pendingAttachments.length, phase, showLanding]);

  const buildHistoryForWorker = useCallback((priorMessages: ChatMessage[]): ChatTurn[] => {
    return priorMessages.map((m) => {
      if (m.role !== "user" || !m.images?.length) {
        return { role: m.role, content: m.content };
      }
      const images: ChatTurn["images"] = [];
      for (const img of m.images) {
        const primary = imageBytesCacheRef.current.get(img.id);
        if (primary?.bytes.byteLength) {
          images.push({ bytes: primary.bytes, mime: primary.mime });
        }
        for (let idx = 1; idx < MAX_ATTACHMENTS * 4; idx++) {
          const extra = imageBytesCacheRef.current.get(`${img.id}:${idx}`);
          if (!extra?.bytes.byteLength) break;
          images.push({ bytes: extra.bytes, mime: extra.mime });
        }
      }
      return { role: m.role, content: m.content, images: images.length ? images : undefined };
    });
  }, []);

  const contextUsage = useMemo((): ContextUsage | null => {
    if (!isLoaded) return null;
    const profileId = loadChatProfileOverride() ?? detectChatHardwareProfile();
    const systemChars =
      measuredSystemPromptCharsRef.current > 0
        ? measuredSystemPromptCharsRef.current
        : defaultSystemPromptChars(appSettings.gemma.systemPrompt, cameraMode);
    const historyBase = cameraMode
      ? buildCameraHistoryForWorker(cameraStore.messages)
      : buildHistoryForWorker(messages);
    const history =
      isGenerating && assistantBuffer.trim()
        ? [...historyBase, { role: "assistant" as const, content: assistantBuffer }]
        : historyBase;
    return estimateLiveContextUsage({
      history,
      draftPrompt: prompt,
      systemPromptChars: systemChars,
      webContextChars: isGenerating ? measuredWebContextCharsRef.current : 0,
      imageCount: pendingAttachments.length + (cameraMode ? 1 : 0),
      profileId,
    });
  }, [
    isLoaded,
    messages,
    cameraStore.messages,
    cameraMode,
    prompt,
    pendingAttachments.length,
    contextRefreshKey,
    appSettings.gemma.systemPrompt,
    buildHistoryForWorker,
    isGenerating,
    assistantBuffer,
  ]);

  const restoredImageBuffersFromMessage = useCallback((msg: ChatMessage): ArrayBuffer[] => {
    if (!msg.images?.length) return [];
    const buffers: ArrayBuffer[] = [];
    for (const img of msg.images) {
      const primary = imageBytesCacheRef.current.get(img.id);
      if (primary?.bytes.byteLength) buffers.push(primary.bytes);
      for (let idx = 1; idx < MAX_ATTACHMENTS * 4; idx++) {
        const extra = imageBytesCacheRef.current.get(`${img.id}:${idx}`);
        if (!extra?.bytes.byteLength) break;
        buffers.push(extra.bytes);
      }
    }
    return buffers;
  }, []);

  const addFilesAsAttachments = useCallback(async (files: FileList | File[]) => {
    setAttachError(null);
    const list = Array.from(files).filter(isAcceptedDocumentFile);
    if (!list.length) {
      setAttachError("פורמט לא נתמך — PDF, תמונה, TXT, DOCX, XLSX, HEIC");
      return;
    }
    const room = MAX_ATTACHMENTS - pendingAttachments.length;
    if (room <= 0) {
      setAttachError(`מקסימום ${MAX_ATTACHMENTS} קבצים`);
      return;
    }
    const slice = list.slice(0, room);
    setAttachProcessing(true);
    try {
      const prepared: PendingAttachment[] = [];
      for (const f of slice) {
        setStatus(`מעבד ${f.name}…`);
        prepared.push(await ingestDocument(f, (msg) => setStatus(msg)));
      }
      setPendingAttachments((prev) => [...prev, ...prepared]);
    } catch (err) {
      setAttachError(err instanceof Error ? err.message : String(err));
    } finally {
      setAttachProcessing(false);
      setStatus("Ready");
    }
  }, [pendingAttachments.length]);

  const removePendingAttachment = useCallback((id: string) => {
    setPendingAttachments((prev) => {
      const target = prev.find((p) => p.id === id);
      if (target) revokePendingAttachment(target);
      return prev.filter((p) => p.id !== id);
    });
  }, []);

  const pushActivity = useCallback((entry: Omit<ModelActivityEntry, "id" | "ts">) => {
    setActivityLog((prev) => appendModelActivity(prev, entry));
  }, []);

  /** Pause vision-lab only while the user chat (text input) is generating — not background scene/HAL calls. */
  const syncVisionBusy = useCallback(() => {
    const userChatBusy = isGeneratingRef.current;
    const runner = cameraLoopRef.current;
    if (!runner) return;
    if (userChatBusy) {
      runner.pauseForChatInference();
    } else if (cameraModeRef.current) {
      runner.resumeAfterChatInference();
    }
  }, []);

  const openArtifact = useCallback((artifact: Artifact) => {
    setActiveArtifact(artifact);
    setArtifactOpen(true);
  }, []);

  const requestSceneAnalysis = useCallback(
    (
      bytes: ArrayBuffer,
      previousSummary: string,
      reason = "scheduled",
      sensorBlock = "",
    ): Promise<SceneAnalysisResult | null> => {
      return new Promise((resolve) => {
        if (!workerRef.current) {
          resolve(null);
          return;
        }
        const requestId = crypto.randomUUID();
        const payload = bytes.slice(0);
        const userPrompt = buildSceneAnalysisUserPrompt(previousSummary, sensorBlock);
        sceneAnalysisMetaRef.current.set(requestId, { reason });
        sceneAnalysisResolversRef.current.set(requestId, resolve);
        workerInferenceBusyRef.current = true;
        pushActivity({
          direction: "out",
          kind: "analyze_scene",
          title: "ניתוח סצנה (Camera)",
          detail: `SYSTEM:\n${SCENE_ANALYSIS_SYSTEM_PROMPT}\n\nUSER:\n${userPrompt}`,
          meta: {
            requestId: requestId.slice(0, 8),
            reason,
            imageKB: Math.round(payload.byteLength / 1024),
            previousSummary: previousSummary ? "yes" : "no",
          },
        });
        workerRef.current.postMessage(
          {
            type: "analyze_scene",
            requestId,
            modelId: GEMMA_MODEL_ID,
            images: [{ bytes: payload, mime: "image/jpeg" }],
            previousSummary,
            sensorBlock,
          },
          [payload],
        );
        window.setTimeout(() => {
          if (sceneAnalysisResolversRef.current.has(requestId)) {
            sceneAnalysisResolversRef.current.delete(requestId);
            sceneAnalysisMetaRef.current.delete(requestId);
            workerInferenceBusyRef.current = false;
            pushActivity({
              direction: "system",
              kind: "analyze_scene",
              title: "Timeout",
              detail: "scene_analysis did not return within 120s",
            });
            resolve(null);
          }
        }, 120_000);
      });
    },
    [pushActivity],
  );

  const requestCharacterUtterance = useCallback(
    (decision: CharacterDecision): Promise<string | null> => {
      return new Promise((resolve) => {
        if (!workerRef.current || isGeneratingRef.current) {
          resolve(null);
          return;
        }
        const requestId = crypto.randomUUID();
        const brain = characterBrainRef.current;
        const world = worldMemoryRef.current;
        const useV2 =
          appSettingsRef.current.vision.vision2Enabled &&
          !!cameraLoopRef.current?.getDialogueContext();
        const sensorBlock = useV2
          ? buildDeepVisionContextBlock(cameraLoopRef.current!.getDialogueContext())
          : buildProactiveSensorBlock(world, poseFromWorld(world), {
              reason: decision.reason,
              topic: decision.topic,
              fallbackHint: decision.message,
            });
        const userPrompt = buildProactiveUserPrompt({
          mood: decision.mood,
          reason: decision.reason,
          topic: decision.topic,
          curiosity: brain.curiosity,
          boredom: brain.boredom,
          sensorBlock,
          fallbackHint: decision.message,
        });
        characterUtteranceResolversRef.current.set(requestId, resolve);
        workerInferenceBusyRef.current = true;
        pushActivity({
          direction: "out",
          kind: "character_utterance",
          title: `Character · ${decision.mood} (Gemma)`,
          detail: `SYSTEM:\n${PROACTIVE_UTTERANCE_SYSTEM}\n\nUSER:\n${userPrompt}`,
          meta: { reason: decision.reason, topic: decision.topic },
        });
        workerRef.current.postMessage({
          type: "character_utterance",
          requestId,
          modelId: GEMMA_MODEL_ID,
          systemPrompt: PROACTIVE_UTTERANCE_SYSTEM,
          userPrompt,
          maxNewTokens: 80,
        });
        window.setTimeout(() => {
          if (characterUtteranceResolversRef.current.has(requestId)) {
            characterUtteranceResolversRef.current.delete(requestId);
            workerInferenceBusyRef.current = false;
            resolve(null);
          }
        }, 30_000);
      });
    },
    [pushActivity],
  );

  const requestSearchPlan = useCallback(
    (query: string, recentUserText: string[]): Promise<string | null> => {
      return new Promise((resolve) => {
        if (!workerRef.current || !isLoaded || isGeneratingRef.current || workerInferenceBusyRef.current) {
          resolve(null);
          return;
        }
        const requestId = crypto.randomUUID();
        const userPrompt = buildSearchPlannerUserPrompt(query, recentUserText);
        searchPlanResolversRef.current.set(requestId, resolve);
        workerInferenceBusyRef.current = true;
        pushActivity({
          direction: "out",
          kind: "web_search",
          title: "Search planner (Gemma)",
          detail: userPrompt.slice(0, 800),
        });
        workerRef.current.postMessage({
          type: "search_plan",
          requestId,
          modelId: GEMMA_MODEL_ID,
          userPrompt,
          maxNewTokens: 120,
        });
        window.setTimeout(() => {
          if (searchPlanResolversRef.current.has(requestId)) {
            searchPlanResolversRef.current.delete(requestId);
            workerInferenceBusyRef.current = false;
            resolve(null);
          }
        }, 18_000);
      });
    },
    [isLoaded, pushActivity],
  );

  const requestGemmaNewsPolish = useCallback(
    (
      articleExcerpt: string,
      articleTitle: string,
      progress?: NewsSummaryGemmaProgress,
    ): Promise<string | null> => {
      return new Promise((resolve) => {
        if (!workerRef.current || !isLoaded) {
          resolve(null);
          return;
        }
        if (workerInferenceBusyRef.current) {
          resolve(null);
          return;
        }
        const requestId = crypto.randomUUID();
        const userPrompt = buildGemmaNewsPolishUserPrompt(articleExcerpt, articleTitle);
        characterUtteranceResolversRef.current.set(requestId, resolve);
        if (progress?.onGemmaToken || progress?.onStreamChunk) {
          characterUtteranceTokenListenersRef.current.set(requestId, {
            onCount: progress.onGemmaToken,
            onChunk: progress.onStreamChunk,
          });
        }
        workerInferenceBusyRef.current = true;
        pushActivity({
          direction: "out",
          kind: "character_utterance",
          title: "סיכום כתבה (Gemma)",
          detail: userPrompt.slice(0, 900),
        });
        workerRef.current.postMessage({
          type: "character_utterance",
          requestId,
          modelId: GEMMA_MODEL_ID,
          systemPrompt: GEMMA_NEWS_POLISH_SYSTEM,
          userPrompt,
          maxNewTokens: 380,
        });
        window.setTimeout(() => {
          if (characterUtteranceResolversRef.current.has(requestId)) {
            characterUtteranceResolversRef.current.delete(requestId);
            characterUtteranceTokenListenersRef.current.delete(requestId);
            workerInferenceBusyRef.current = false;
            resolve(null);
          }
        }, 120_000);
      });
    },
    [isLoaded, pushActivity],
  );

  const requestOpenWebQueryPlan = useCallback(
    (query: string, ruleQueries: string[]): Promise<string | null> => {
      return new Promise((resolve) => {
        if (!workerRef.current || !isLoaded || workerInferenceBusyRef.current) {
          resolve(null);
          return;
        }
        const requestId = crypto.randomUUID();
        const userPrompt = buildOpenWebPlannerUserPrompt(query, ruleQueries);
        characterUtteranceResolversRef.current.set(requestId, resolve);
        workerInferenceBusyRef.current = true;
        pushActivity({
          direction: "out",
          kind: "web_search",
          title: "Open-web query planner (Gemma)",
          detail: userPrompt.slice(0, 800),
        });
        workerRef.current.postMessage({
          type: "character_utterance",
          requestId,
          modelId: GEMMA_MODEL_ID,
          systemPrompt: OPEN_WEB_QUERY_PLANNER_SYSTEM,
          userPrompt,
          maxNewTokens: 120,
        });
        window.setTimeout(() => {
          if (characterUtteranceResolversRef.current.has(requestId)) {
            characterUtteranceResolversRef.current.delete(requestId);
            workerInferenceBusyRef.current = false;
            resolve(null);
          }
        }, 18_000);
      });
    },
    [isLoaded, pushActivity],
  );

  const resolveSearchPlanForQuery = useCallback(
    async (query: string, recentUserText: string[]): Promise<SearchPlan | undefined> => {
      const topicPlan = buildWebTopicSearchPlan(query);
      if (topicPlan?.kind === "cinema_il") {
        return planToSearchPlan(topicPlan);
      }
      if (topicPlan) {
        const raw = await requestOpenWebQueryPlan(query, topicPlan.engineQueries);
        const queries = parseOpenWebQueriesJson(raw, topicPlan.engineQueries);
        return {
          ...planToSearchPlan({ ...topicPlan, engineQueries: queries }),
          reason: `open-web:${topicPlan.kind}${raw ? "+gemma" : ""}`,
        };
      }
      const regexPlan = regexPlanForQuery(query);
      if (regexPlan) return regexPlan;
      if (!shouldUseSearchPlanner(query)) return undefined;
      const raw = await requestSearchPlan(query, recentUserText);
      if (!raw) return undefined;
      return parseSearchPlanJson(raw, query) ?? undefined;
    },
    [requestOpenWebQueryPlan, requestSearchPlan],
  );

  const stopCameraMode = useCallback(() => {
    cameraBootingRef.current = false;
    cameraLoopRef.current?.dispose();
    cameraLoopRef.current = null;
    mountGroveeVisionProbe(null);
    cameraStreamRef.current?.stop();
    cameraStreamRef.current = null;
    setCameraStream(null);
    worldMemoryRef.current.reset();
    persistCameraMemory();
    setCharacterMood("observing");
    setCameraMode(false);
    setCameraObserving(false);
    setCameraError(null);
    setCameraStatus("");
    sceneAnalysisResolversRef.current.forEach((resolve) => resolve(null));
    sceneAnalysisResolversRef.current.clear();
    characterUtteranceResolversRef.current.forEach((resolve) => resolve(null));
    characterUtteranceResolversRef.current.clear();
    characterUtteranceTokenListenersRef.current.clear();
    searchPlanResolversRef.current.forEach((resolve) => resolve(null));
    searchPlanResolversRef.current.clear();
  }, [persistCameraMemory]);

  const toggleCameraMode = useCallback(async () => {
    if (cameraMode) {
      stopCameraMode();
      return;
    }
    if ((!isGemmaLoaded && !QA_VISION_MODE) || isGenerating) return;
    setCameraError(null);
    const support = checkBrowserVisionSupport();
    if (!support.ok) {
      setCameraError(support.message ?? "מצב מצלמה לא נתמך בדפדפן זה");
      return;
    }
    try {
      const handle = await requestCameraStream();
      cameraStreamRef.current = handle;
      setCameraStream(handle.stream);
      setCameraMode(true);
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : String(err));
      setCameraMode(false);
      setCameraStream(null);
    }
  }, [cameraMode, isGenerating, isGemmaLoaded, stopCameraMode]);

  useEffect(() => {
    if (!QA_VISION_MODE || cameraModeRef.current) return;
    void toggleCameraMode();
  }, [toggleCameraMode]);

  useEffect(() => {
    if (!QA_VISION_MODE || !visionResult) return;
    setVisionInspectorOpen(true);
  }, [visionResult]);

  const startVisionPipeline = useCallback(
    async (video: HTMLVideoElement) => {
      if (!cameraStream || (!isGemmaLoaded && !QA_VISION_MODE) || cameraBootingRef.current || cameraLoopRef.current) return;
      cameraBootingRef.current = true;
      try {
        await attachStreamToVideo(video, cameraStream);

        worldMemoryRef.current.reset();
        characterBrainRef.current.importSnapshot(cameraStoreRef.current.memory);
        chatTopicRef.current = null;

        const visionBehavior = appSettingsRef.current.vision;
        const visionBudget = {
          ...detectVisionBudget(),
          useLlmDeepVision:
            visionBehavior.useLlmDeepVision && detectVisionBudget().useLlmDeepVision,
          useLlmProactiveUtterance:
            visionBehavior.useLlmProactiveUtterance && detectVisionBudget().tier === "normal",
        };
        cameraVisionBudgetRef.current = visionBudget;
        const perfMode = visionBehavior.performanceMode;
        const labConfig = ensureVisionLabConfig(
          {
            ...visionPipelineConfig,
            performanceMode: perfMode,
            sampleIntervals: intervalsFromMode(perfMode),
          },
          visionBudget.tier === "low" ? "low" : "normal",
        );
        setVisionPipelineConfig(labConfig);

        pushActivity({
          direction: "system",
          kind: "camera_loop",
          title: "Vision Lab · boot",
          detail: `tier=${visionBudget.tier}\nhands=${labConfig.toggles.hands}\npose=${labConfig.toggles.pose}\nyolo=${labConfig.toggles.yolo}\nhandsMs=${labConfig.sampleIntervals.hands}`,
        });

        const runner = new GroveeVisionRunner(
          worldMemoryRef.current,
          characterBrainRef.current,
          {
            requestAnalysis: async (req) =>
              requestSceneAnalysis(req.bytes, req.previousSummary, req.reason, req.sensorBlock ?? ""),
            useLlmProactiveUtterance: () => {
              const b = cameraVisionBudgetRef.current;
              if (!appSettingsRef.current.vision.useLlmProactiveUtterance) return false;
              if (!b.useLlmProactiveUtterance) return false;
              if (cameraLoopRef.current?.isDeepVisionDegraded()) return false;
              return true;
            },
            useBootDeepSnapshot: () =>
              QA_VISION_MODE ? false : appSettingsRef.current.vision.useBootDeepSnapshot,
            resolveUtterance: async (decision) => {
              const llm = await requestCharacterUtterance(decision);
              if (llm?.trim()) return { ...decision, message: llm.trim() };
              return decision;
            },
            isWorkerBusy: () => isGeneratingRef.current || workerInferenceBusyRef.current,
            onCameraStatus: setCameraStatus,
            onMoodChange: setCharacterMood,
            onHalStateUpdate: setHalMoodState,
            onPipelineProgress: setVisionPipelineProgress,
            onVisionResult: (result) => {
              visionResultRef.current = result;
              setVisionResult({ ...result });
              setWorldMemorySnapshot(worldMemoryRef.current.toInspectorSnapshot());
            },
            onCharacterSpeak: (decision: CharacterDecision) => {
              pushActivity({
                direction: "in",
                kind: "character_speak",
                title: `Character · ${decision.mood}`,
                detail: `${decision.reason}\n\n${decision.message}`,
                meta: { topic: decision.topic, mood: decision.mood },
              });
              if (!cameraModeRef.current) return;
              setCameraStore((prev) => {
                const topicLog = appendTopicToLog(prev.memory.topicLog, decision.topic);
                const next: CameraSessionStore = {
                  ...prev,
                  updatedAt: Date.now(),
                  memory: { ...prev.memory, topicLog },
                  messages: [
                    ...prev.messages,
                    {
                      id: crypto.randomUUID(),
                      role: "assistant",
                      kind: "proactive",
                      content: decision.message,
                      ts: Date.now(),
                      modelLabel: `HAL · ${moodLabelHe(decision.mood)}`,
                    },
                  ],
                };
                saveCameraSessionStore(next);
                return next;
              });
            },
            onObservingChange: setCameraObserving,
            onLightDetection: (payload) => {
              if (!appSettingsRef.current.vision.logVisionToActivity) return;
              const evLines = payload.worldUpdate.newEvents.map(
                (e) => `[${e.type}] ${e.text}${e.subject ? ` (${e.subject})` : ""}`,
              );
              const c = payload.consciousness;
              pushActivity({
                direction: "system",
                kind: "light_detect",
                title: c ? "HAL · Consciousness" : "Vision Lab · YOLO",
                detail: [
                  c
                    ? `STATE: ${c.soul} (${(c.confidence * 100).toFixed(0)}%)`
                    : `personCount(raw): ${payload.rawPeople}`,
                  c ? `person STABLE: ${c.personStable ? "YES" : "NO"}` : "",
                  c ? `raw sensor: ${c.rawDetected ? "detected" : "none"} (ignore alone)` : "",
                  c ? `interpretation: ${c.interpretation}` : "",
                  !c ? `people(debounced): ${payload.debouncedPeople.join(", ") || "(none)"}` : "",
                  `objects: ${payload.objects.join(", ") || "(none)"}`,
                  payload.personJustConfirmed ? "transition: became STABLE" : "",
                  payload.personJustLeft ? "transition: PRESENCE_COLLAPSE" : "",
                  payload.worldUpdate.isBaselineCapture ? "baseline capture (no events)" : "",
                  payload.worldUpdate.suppressedAsChurn ? "suppressed: camera churn" : "",
                  evLines.length ? `events:\n${evLines.join("\n")}` : "",
                ]
                  .filter(Boolean)
                  .join("\n"),
                meta: {
                  personCount: payload.rawPeople,
                  personConfirmed: c ? c.personStable : payload.debouncedPeople.length > 0,
                  eventCount: payload.worldUpdate.newEvents.length,
                },
              });
            },
            onSituationUpdate: (payload) => {
              pushActivity({
                direction: "system",
                kind: "situation",
                title: "Vision Lab · Pose + Gestures",
                detail: [
                  `source: ${payload.poseSource}`,
                  payload.poseConfidence !== undefined
                    ? `confidence: ${payload.poseConfidence.toFixed(2)}`
                    : "",
                  `pose: ${payload.poseState}`,
                  payload.gestures.length ? `gestures: ${payload.gestures.join(", ")}` : "",
                  payload.holding.length ? `holding: ${payload.holding.join(", ")}` : "",
                  payload.focusHint ? `focus: ${payload.focusHint}` : "",
                  payload.newEvents.length
                    ? `events: ${payload.newEvents.map((e) => e.subject ?? e.type).join(", ")}`
                    : "",
                ]
                  .filter(Boolean)
                  .join("\n"),
              });
            },
            onLabEvents: (events) => {
              if (!events.length) return;
              pushActivity({
                direction: "system",
                kind: "camera_loop",
                title: "Vision Lab · Events",
                detail: events.map((e) => `${e.text} (${e.subject ?? e.type})`).join("\n"),
              });
            },
            useVision2: () => appSettingsRef.current.vision.vision2Enabled,
            onVision2Update: ({ snapshot, dialogue }) => {
              setHalInterpretation(dialogue.interpretation ?? null);
              setHalConsciousness(dialogue.consciousness ?? null);
              setHalEntity(dialogue.entity ?? null);
              if (!appSettingsRef.current.vision.logVisionToActivity) return;
              const c = dialogue.consciousness;
              pushActivity({
                direction: "system",
                kind: "vision2",
                title: "HAL · Perception",
                detail: [
                  c
                    ? `SOUL: ${c.soul} → evolution: ${c.evolution}`
                    : `person: ${snapshot.person.present ? "STABLE" : "absent"}`,
                  c ? `confidence: ${(c.confidence * 100).toFixed(0)}% · stable ${c.stabilitySec.toFixed(1)}s` : "",
                  c ? `affect: curiosity ${(c.affect.curiosity * 100).toFixed(0)}% certainty ${(c.affect.certainty * 100).toFixed(0)}%` : "",
                  `scene: ${dialogue.interpretation?.sceneState.activity ?? "?"} / ${dialogue.interpretation?.sceneState.stability ?? "?"}`,
                  `meta: ${dialogue.interpretation?.metaEvents[0]?.type ?? "none"}`,
                  `situation: ${snapshot.situation.primary} (${(snapshot.situation.confidence * 100).toFixed(0)}%)`,
                  `hal: ${dialogue.hal.mood} / ${dialogue.hal.tone}${dialogue.hal.sceneLabel ? ` / ${dialogue.hal.sceneLabel}` : ""}`,
                  `body: focused=${snapshot.bodyLanguage.focused.toFixed(2)} thinking=${snapshot.bodyLanguage.thinking.toFixed(2)} stressed=${snapshot.bodyLanguage.stressed.toFixed(2)}`,
                  dialogue.capabilities.productivity.needsBreak
                    ? "productivity: break suggested"
                    : "",
                  dialogue.capabilities.teaching.likelyDistracted
                    ? `teaching: attention loss ${dialogue.capabilities.teaching.attentionLoss.toFixed(2)}`
                    : "",
                  dialogue.coach.intent !== "none"
                    ? `coach: ${dialogue.coach.intent} — ${dialogue.coach.reason}`
                    : "",
                ]
                  .filter(Boolean)
                  .join("\n"),
              });
            },
          },
          visionBudget,
          labConfig,
        );
        runner.bindVideo(video);
        cameraLoopRef.current = runner;
        mountGroveeVisionProbe(runner);
        await runner.start();
        setCameraStatus("👁 Character · Vision Lab פעיל");
      } catch (err) {
        setCameraError(err instanceof Error ? err.message : String(err));
        stopCameraMode();
      } finally {
        cameraBootingRef.current = false;
      }
    },
    [
      cameraStream,
      isLoaded,
      visionPipelineConfig,
      requestSceneAnalysis,
      requestCharacterUtterance,
      setMessages,
      stopCameraMode,
      pushActivity,
    ],
  );

  useEffect(() => {
    if (!cameraMode || !cameraStream || (!isGemmaLoaded && !QA_VISION_MODE)) return;
    const video = cameraVideoRef.current;
    if (!video || cameraLoopRef.current || cameraBootingRef.current) return;
    void startVisionPipeline(video);
  }, [cameraMode, cameraStream, isLoaded, startVisionPipeline]);

  useEffect(() => {
    return () => {
      cameraLoopRef.current?.dispose();
      cameraStreamRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    if (phase !== "ready" && cameraMode && !QA_VISION_MODE) stopCameraMode();
  }, [phase, cameraMode, stopCameraMode]);

  useEffect(() => {
    if (cameraMode && !isGenerating) {
      syncVisionBusy();
    }
  }, [cameraMode, isGenerating, syncVisionBusy]);

  const onCameraPipelineConfigChange = useCallback((partial: Partial<PipelineConfig>) => {
    setVisionPipelineConfig((prev) => {
      const next = {
        ...prev,
        ...partial,
        toggles: { ...prev.toggles, ...partial.toggles },
        sampleIntervals: { ...prev.sampleIntervals, ...partial.sampleIntervals },
      };
      cameraLoopRef.current?.applyPipelineConfig(next);
      savePipelineConfig(next);
      return next;
    });
  }, []);

  const showSearchResultsPanel =
    searchResultsOpen && !!searchResultsPayload && !gamesPanelOpen && !liveMediaPanelOpen;
  const showLiveMediaPanel = liveMediaPanelOpen && !showSearchResultsPanel;
  const showArtifactPanel =
    artifactOpen &&
    !!activeArtifact &&
    !gamesPanelOpen &&
    !globePanelOpen &&
    !showSearchResultsPanel &&
    !showLiveMediaPanel;
  const showGamesPanel =
    gamesPanelOpen &&
    desktopLayout &&
    !globePanelOpen &&
    !showSearchResultsPanel &&
    !showLiveMediaPanel;
  const showGamesFullscreen = showGamesPanel && gamesPanelLayout === "full";
  const showGamesSidePanel = showGamesPanel && gamesPanelLayout === "side";
  const showGlobePanel = globePanelOpen && !showSearchResultsPanel && !showLiveMediaPanel;
  const showCameraSidePanel =
    cameraMode &&
    desktopLayout &&
    !showArtifactPanel &&
    !gamesPanelOpen &&
    !globePanelOpen &&
    !showSearchResultsPanel &&
    !showLiveMediaPanel;
  const showCameraInline = cameraMode && !desktopLayout;
  const showLiveMediaFullscreen = showLiveMediaPanel && desktopLayout;
  const sidePanelBesideChat =
    showArtifactPanel ||
    showCameraSidePanel ||
    showGamesSidePanel ||
    showGlobePanel ||
    showSearchResultsPanel ||
    (showLiveMediaPanel && !desktopLayout);
  const anySidePanelOpen = sidePanelBesideChat || showLiveMediaFullscreen || showGamesFullscreen;

  const closeSearchResultsPanel = useCallback(() => {
    setSearchResultsOpen(false);
    setSearchResultsPayload(null);
    clearSearchResultsPayload();
  }, []);

  const handleSearchPanelQuery = useCallback(
    async (query: string) => {
      const q = query.trim();
      if (!q) return;
      setSearchPanelLoading(true);
      setStatus("מחפש מידע…");
      try {
        const recentUserText = messages
          .filter((m) => m.role === "user")
          .slice(-4)
          .map((m) => m.content);
        const searchPlan = await resolveSearchPlanForQuery(q, recentUserText);
        const panelPlan = buildPanelSearchPlan(q);
        const planForSearch = searchPlan
          ? {
              queries: searchPlan.queries.length ? searchPlan.queries : panelPlan.queries,
              answerShape: searchPlan.answerShape ?? panelPlan.answerShape,
              useWebFallback: searchPlan.useWebFallback ?? true,
              blendNewsWithWeb: searchPlan.blendNewsWithWeb ?? false,
            }
          : panelPlan;
        const liveSources: SearchSourceResult[] = [];
        const searchResult = await runWebSearch(q, {
          recentUserText,
          panelSearch: true,
          plan: planForSearch,
          onProgress: (ev) => {
            if (ev.type === "provider_done") {
              const rest = liveSources.filter((s) => s.provider !== ev.result.provider);
              liveSources.length = 0;
              liveSources.push(...rest, ev.result);
              if (ev.result.provider === "grovee-news" && (ev.result.newsCards?.length ?? 0) > 0) {
                setSearchResultsPayload(buildUnifiedSearchPayload(q, [...liveSources]));
                setSearchResultsOpen(true);
              }
              if (ev.result.provider === "huggingface-models" && (ev.result.hfModelHits?.length ?? 0) > 0) {
                setSearchResultsPayload(buildUnifiedSearchPayload(q, [...liveSources]));
                setSearchResultsOpen(true);
              }
            }
            if (ev.type === "complete") {
              liveSources.length = 0;
              liveSources.push(...ev.sources);
            }
          },
        });
        const unifiedSearchPayload = buildUnifiedSearchPayload(q, searchResult.sources);
        setSearchResultsPayload(unifiedSearchPayload);
        setSearchResultsOpen(true);
        setArtifactOpen(false);
        setGlobePanelOpen(false);
        setGamesPanelOpen(false);
        setGamesEmbedGame(null);
      } finally {
        setSearchPanelLoading(false);
        setStatus("");
      }
    },
    [messages, resolveSearchPlanForQuery],
  );

  const openPluginsHub = useCallback((tab: PluginsHubTab = "plugins") => {
    setPluginsHubTab(tab);
    setPluginsOpen(true);
  }, []);

  const handleGearMenuAction = useCallback(
    (action: SidebarGearAction) => {
      switch (action) {
        case "settings":
          setSettingsInitialTab("gemma");
          setSettingsModalKey((k) => k + 1);
          setSettingsOpen(true);
          break;
        case "api-keys":
          setSettingsInitialTab("api-keys");
          setSettingsModalKey((k) => k + 1);
          setSettingsOpen(true);
          break;
        case "plugins":
          openPluginsHub("plugins");
          break;
        case "activity":
          setActivityLogOpen(true);
          break;
        case "presentation-qa":
          setPresentationQaOpen(true);
          break;
        case "vision":
          setVisionInspectorOpen(true);
          break;
      }
    },
    [openPluginsHub],
  );

  const handleRackModelSelect = useCallback((id: string) => {
    setSelectedRackModelId(id);
    persistSelectedModelId(id);
  }, []);

  const handleDownloadLocalText = useCallback(
    async (entry: RackModelEntry) => {
      if (!entry.hfModelId || localTextDownloadingId) return;
      setLocalTextDownloadingId(entry.id);
      setLocalTextDownloadPct(0);
      setLocalTextDownloadLabel("מתחיל הורדה…");
      try {
        await downloadLocalTextModel(
          entry.id,
          entry.hfModelId,
          (p) => {
            setLocalTextDownloadPct(p.pct);
            setLocalTextDownloadLabel(p.message);
          },
          appSettingsRef.current.localText.inferenceBackend,
        );
        setModelRack(loadModelRack());
        setStatus(`${entry.label} מוכן לשיחה`);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setStatus(`שגיאת הורדה: ${msg}`);
      } finally {
        setLocalTextDownloadingId(null);
        setLocalTextDownloadPct(0);
        setLocalTextDownloadLabel("");
      }
    },
    [localTextDownloadingId],
  );

  const pickerModelRack = useMemo(
    () => applyLocalTextDownloadStates(modelRack, localTextDownloadingId),
    [modelRack, localTextDownloadingId],
  );

  const handleRackUpdated = useCallback((_rack?: RackModelEntry[]) => {
    setModelRack(loadModelRack());
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const rack = await refreshCloudModelRack((p) => {
          if (p.phase !== "cloud-done") {
            setStatus(`בודק מודלי תמונה: ${p.phase} (${p.cloud ?? p.found} פעילים)…`);
          }
        });
        if (cancelled) return;
        setModelRack(rack);
        if (
          !rack.some(
            (r) => r.id === selectedRackModelRef.current && isSelectableInPicker(r),
          )
        ) {
          persistSelectedModelId(GEMMA_RACK_ID);
          setSelectedRackModelId(GEMMA_RACK_ID);
        }
        const counts = summarizeRackCounts(rack);
        setStatus(counts.cloud > 0 ? `Ready · GroVee + ${counts.cloud} תמונה` : "Ready");
      } catch {
        if (!cancelled) setModelRack(loadModelRack());
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const openSearchPanelFull = useCallback(() => {
    setSearchResultsPayload(createEmptySearchPayload());
    setSearchResultsOpen(true);
    setArtifactOpen(false);
    setGlobePanelOpen(false);
    setGamesPanelOpen(false);
    setLiveMediaPanelOpen(false);
    setGamesEmbedGame(null);
  }, []);

  const openLiveMediaPanelFull = useCallback(() => {
    setLiveMediaPanelOpen(true);
    setSearchResultsOpen(false);
    setSearchResultsPayload(null);
    setArtifactOpen(false);
    setGlobePanelOpen(false);
    setGamesPanelOpen(false);
    setGamesEmbedGame(null);
  }, []);

  const toggleLiveMediaPanel = useCallback(() => {
    if (liveMediaPanelOpen) {
      setLiveMediaPanelOpen(false);
      return;
    }
    openLiveMediaPanelFull();
  }, [liveMediaPanelOpen, openLiveMediaPanelFull]);

  const closeLiveMediaPanel = useCallback(() => {
    setLiveMediaPanelOpen(false);
  }, []);

  const handlePlayGame = useCallback((game: OnlineGame) => {
    setGamesPanelLayout("full");
    setGamesEmbedGame(game);
    setGamesPanelOpen(true);
    setArtifactOpen(false);
    void recordGamePlay(game);
  }, []);

  const handleGamesPanelUpdate = useCallback(
    (games: OnlineGame[], title?: string, category?: GameCategoryId | null) => {
      setGamesPanelGames(games);
      if (title) setGamesPanelTitle(title);
      if (category !== undefined) setGamesPanelCategory(category ?? "featured");
      if (games.length > 0) {
        void saveGamesSession(games, title ?? gamesPanelTitle, category ?? gamesPanelCategory);
      }
    },
    [gamesPanelCategory, gamesPanelTitle],
  );

  const closeGamesPanel = useCallback(() => {
    setGamesPanelOpen(false);
    setGamesEmbedGame(null);
    setGamesPanelLayout("side");
  }, []);

  const expandGamesPanelFull = useCallback(() => {
    setGamesPanelLayout("full");
  }, []);

  const shrinkGamesPanelSide = useCallback(() => {
    setGamesPanelLayout("side");
  }, []);

  const openGlobePanelFull = useCallback(() => {
    setGlobePanelOpen(true);
    setGamesPanelOpen(false);
    setArtifactOpen(false);
    setLiveMediaPanelOpen(false);
    setGamesEmbedGame(null);
  }, []);

  const closeGlobePanel = useCallback(() => {
    setGlobePanelOpen(false);
  }, []);

  const openGamesPanelFull = useCallback(async () => {
    setGamesPanelLayout("full");
    setGamesPanelOpen(true);
    setGlobePanelOpen(false);
    setArtifactOpen(false);
    setLiveMediaPanelOpen(false);
    setGamesEmbedGame(null);
    setGamesPanelStartView("browse");
    setGamesPanelCategory("featured");
    setGamesPanelLoading(true);
    try {
      const cached = await loadGamesSession();
      const offline = typeof navigator !== "undefined" && !navigator.onLine;
      if (offline && cached?.games.length) {
        setGamesPanelGames(cached.games);
        setGamesPanelTitle(`${cached.title} (מקומי)`);
        return;
      }
      const result = await randomOnlineGames(GAMES_CATALOG_PAGE_SIZE, "featured");
      setGamesPanelGames(result.games);
      setGamesPanelTitle(categoryLabelHe("featured"));
      void saveGamesSession(result.games, categoryLabelHe("featured"), "featured");
    } finally {
      setGamesPanelLoading(false);
    }
  }, []);

  const openGamesFavorites = useCallback(() => {
    setGamesPanelLayout("side");
    setGamesPanelOpen(true);
    setGlobePanelOpen(false);
    setArtifactOpen(false);
    setLiveMediaPanelOpen(false);
    setGamesEmbedGame(null);
    setGamesPanelStartView("favorites");
  }, []);

  const handleGameCategoryPick = useCallback(async (cat: GameCategoryId) => {
    setGamesPanelLayout("side");
    setGamesPanelStartView("browse");
    setGamesPanelCategory(cat);
    setGamesPanelOpen(true);
    setArtifactOpen(false);
    setGamesEmbedGame(null);
    setGamesPanelLoading(true);
    try {
      const result = await randomOnlineGames(12, cat);
      setGamesPanelGames(result.games);
      setGamesPanelTitle(categoryLabelHe(cat));
      void saveGamesSession(result.games, categoryLabelHe(cat), cat);
    } finally {
      setGamesPanelLoading(false);
    }
  }, []);

  const applyLandingSuggestion = useCallback((text: string) => {
    setPrompt(text);
    focusComposerInput();
  }, [focusComposerInput]);

  const appendVoiceTranscript = useCallback((text: string) => {
    const chunk = text.trim();
    if (!chunk) return;
    setPrompt((prev) => (prev.trim() ? `${prev.trimEnd()} ${chunk}` : chunk));
    focusComposerInput();
  }, [focusComposerInput]);

  const finalizeAssistantReply = useCallback(
    (stopped: boolean) => {
      isGeneratingRef.current = false;
      setIsGenerating(false);
      syncVisionBusy();
      const raw = assistantBufferRef.current;
      const visionContext = lastTurnUsedVisionRef.current
        ? pendingVisionContextRef.current || undefined
        : undefined;
      const searchMeta = pendingWebSearchRef.current ?? undefined;
      const timeWidget = pendingTimeWidgetRef.current ?? undefined;
      const showGameCategories = pendingGameCategoryPickerRef.current;
      const gameBrowseCategory = pendingGameBrowseCategoryRef.current ?? undefined;
      const { content, artifact, thought } = raw.trim()
        ? buildPersistedAssistantPayload(raw, thinkingRef.current, {
            chatOnlyDocument: generationChatOnlyDocumentRef.current,
          })
        : { content: "", artifact: null, thought: undefined };

      let finalContent = content;
      if (
        hasPlaceholderReply(finalContent) &&
        searchMeta?.sources?.length &&
        searchMeta.query &&
        needsOpenWebEnrichment(searchMeta.query)
      ) {
        const fallback =
          buildOpenWebTopicReply(searchMeta.query, searchMeta.sources) ??
          buildCapabilityLiveReply(searchMeta.query, [], searchMeta.sources);
        if (fallback?.trim()) finalContent = fallback;
      }

      if (
        !finalContent.trim() &&
        !artifact &&
        !showGameCategories &&
        !timeWidget &&
        searchMeta?.sources?.length &&
        searchMeta.query
      ) {
        const cannedFallback = buildCapabilityLiveReply(
          searchMeta.query,
          [],
          searchMeta.sources,
          { answerShape: searchMeta.answerShape },
        );
        if (cannedFallback?.trim()) finalContent = cannedFallback;
      }

      if (!finalContent.trim() && !artifact && !showGameCategories && !timeWidget) {
        setAssistantBuffer("");
        assistantBufferRef.current = "";
        pendingVisionContextRef.current = "";
        pendingWebSearchRef.current = null;
        pendingTimeWidgetRef.current = null;
        pendingGameCategoryPickerRef.current = false;
        pendingGameBrowseCategoryRef.current = null;
        setStreamingSearchSources(null);
        setStreamingGameCategoryPicker(false);
        setStatus(stopped ? "התשובה נעצרה" : "Ready");
      continueModeRef.current = false;
      cameraLoopRef.current?.releaseAfterChat();
      qaTurnForceLlmRef.current = false;
      focusComposerInput();
        qaChatBridge.notifyTurnFailed(stopped ? "stopped empty" : "empty reply");
        return;
      }

      if (continueModeRef.current) {
        continueModeRef.current = false;
        if (generationCameraModeRef.current) {
          setCameraMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i--) {
              if (next[i].role === "assistant") {
                next[i] = {
                  ...next[i],
                  content: finalContent,
                  thought: thought ?? next[i].thought,
                  visionContext: visionContext ?? next[i].visionContext,
                };
                return next;
              }
            }
            return [
              ...next,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                kind: "reply",
                content: finalContent,
                ts: Date.now(),
                thought,
                visionContext,
                modelLabel: "HAL",
              },
            ];
          });
        } else {
          setMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i--) {
              if (next[i].role === "assistant") {
                next[i] = {
                  ...next[i],
                  content: artifact ? next[i].content || finalContent : finalContent,
                  artifact: artifact ?? next[i].artifact,
                  thought: thought ?? next[i].thought,
                  visionContext: visionContext ?? next[i].visionContext,
                };
                return next;
              }
            }
            return [
              ...next,
              {
                id: crypto.randomUUID(),
                role: "assistant" as const,
                content: finalContent,
                artifact: artifact ?? undefined,
                thought,
                visionContext,
                searchSources: searchMeta?.sources,
                searchSummary: searchMeta?.summary,
                timeWidget,
                showGameCategories,
                gameBrowseCategory,
                modelLabel: "HAL",
              },
            ];
          });
        }
      } else if (generationCameraModeRef.current) {
        appendCameraAssistantMessage({
          content: finalContent,
          kind: "reply",
          modelLabel: "HAL",
          thought,
          visionContext,
        });
        setCameraStore((prev) => {
          const next = {
            ...prev,
            rollingSummary: buildRollingSummary(prev.messages),
            updatedAt: Date.now(),
          };
          saveCameraSessionStore(next);
          return next;
        });
        persistCameraMemory();
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: finalContent,
            artifact: artifact ?? undefined,
            thought,
            visionContext,
            searchSources: searchMeta?.sources,
            searchSummary: searchMeta?.summary,
            timeWidget,
            showGameCategories,
            gameBrowseCategory,
            modelLabel: "HAL",
          },
        ]);
      }

      if (artifact && !generationChatOnlyDocumentRef.current) {
        setActiveArtifact(artifact);
      }

      setAssistantBuffer("");
      assistantBufferRef.current = "";
      pendingVisionContextRef.current = "";
      pendingWebSearchRef.current = null;
      pendingTimeWidgetRef.current = null;
      pendingGameCategoryPickerRef.current = false;
      pendingGameBrowseCategoryRef.current = null;
      setStreamingSearchSources(null);
      setStreamingGameCategoryPicker(false);
      setStreamingVisionContext("");
      setStatus(stopped ? "התשובה נעצרה" : "Ready");
      generationCameraModeRef.current = false;
      generationChatOnlyDocumentRef.current = false;
      setChatOnlyDocumentMode(false);
      cameraLoopRef.current?.releaseAfterChat();
      qaTurnForceLlmRef.current = false;
      focusComposerInput();
      qaChatBridge.notifyTurnComplete(
        finalContent,
        searchMeta?.summary,
        searchMeta?.sources?.filter((s) => s.ok).map((s) => s.provider),
      );
    },
    [setMessages, setCameraMessages, appendCameraAssistantMessage, persistCameraMemory, syncVisionBusy, focusComposerInput],
  );

  const handleNewsArticlePolish = useCallback(
    async (
      card: GroveeNewsCard,
      gemmaInput: string,
      progress?: NewsSummaryGemmaProgress,
    ): Promise<string> => {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: "סכם כתבה",
        },
      ]);

      if (!isLoaded) {
        const err = GEMMA_SUMMARY_FALLBACK_HE;
        assistantBufferRef.current = err;
        setAssistantBuffer(err);
        finalizeAssistantReply(false);
        return err;
      }

      isGeneratingRef.current = true;
      setIsGenerating(true);
      assistantBufferRef.current = "";
      setAssistantBuffer("");
      setStatus("מנסח תקציר בעברית…");

      const polished = await requestGemmaNewsPolish(gemmaInput, card.title, {
        onGemmaToken: progress?.onGemmaToken,
        onStreamChunk: (chunk) => {
          setAssistantBuffer((prev) => {
            const next = prev + chunk;
            assistantBufferRef.current = next;
            return next;
          });
        },
      });

      const final = finalizeGemmaNewsSummary(polished);

      assistantBufferRef.current = final;
      setAssistantBuffer(final);
      pushActivity({
        direction: "in",
        kind: "character_utterance",
        title: "סיכום כתבה בצ'אט",
        detail: final.slice(0, 1200),
        meta: { source: card.source, url: card.url },
      });
      finalizeAssistantReply(false);
      return final;
    },
    [isLoaded, requestGemmaNewsPolish, pushActivity, setMessages, finalizeAssistantReply],
  );

  const stopGeneration = useCallback(() => {
    if (!isGenerating) return;
    workerRef.current?.postMessage({ type: "abort" });
    abortLocalTextGeneration();
    setStatus("עוצר…");
  }, [isGenerating]);

  /** Watchdog: if the worker goes silent mid-generation (GPU crash, dead worker),
   *  recover the UI instead of leaving the chat stuck and ignoring new messages. */
  useEffect(() => {
    if (!isGenerating) return;
    lastChatSignalRef.current = Date.now();
    const STALL_MS = 180_000;
    const timer = window.setInterval(() => {
      if (!isGeneratingRef.current) return;
      if (Date.now() - lastChatSignalRef.current < STALL_MS) return;
      workerRef.current?.postMessage({ type: "abort" });
      pushActivity({
        direction: "in",
        kind: "error",
        title: "Watchdog · יצירה נתקעה",
        detail: `אין אות מה-worker מעל ${Math.round(STALL_MS / 1000)} שניות — משחרר את הצ'אט`,
      });
      if (!assistantBufferRef.current.trim()) {
        const recoveryMsg =
          "⚠️ התשובה נתקעה ולא הושלמה (ייתכן עומס זיכרון GPU). נסה לשלוח שוב; אם זה חוזר — עבור ל-WASM בהגדרות או פתח שיחה חדשה.";
        assistantBufferRef.current = recoveryMsg;
        setAssistantBuffer(recoveryMsg);
      }
      finalizeAssistantReply(true);
      setStatus("היצירה נתקעה ושוחררה — אפשר לשלוח שוב");
    }, 5_000);
    return () => window.clearInterval(timer);
  }, [isGenerating, finalizeAssistantReply, pushActivity]);

  useEffect(() => {
    if (phase !== "ready") return;
    if (generationChatOnlyDocumentRef.current) {
      return;
    }
    const streamSource = getArtifactScanContent(assistantBuffer, thinkingRef.current).trim();
    if (streamSource) {
      const detected = extractPrimaryArtifact(streamSource);
      if (detected) {
        setActiveArtifact(detected);
        setArtifactOpen(true);
      }
      return;
    }
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    if (!lastAssistant) {
      setActiveArtifact(null);
      return;
    }
    if (lastAssistant.artifact) {
      setActiveArtifact(lastAssistant.artifact);
      return;
    }
    const detected = extractPrimaryArtifact(
      getArtifactScanContent(lastAssistant.content, false).trim() || lastAssistant.content,
    );
    setActiveArtifact(detected);
  }, [assistantBuffer, messages, phase]);

  const runGemmaGenerate = (
    promptText: string,
    history: ChatTurn[],
    systemPrompt: string,
    maxNewTokens: number,
    temperature: number,
    repetitionPenalty: number,
    topP: number,
    webContext: string,
    currentImages: ArrayBuffer[] = [],
    currentImageMime = "image/jpeg",
  ) => {
    const historyForWorker = history.map((turn) => ({
      ...turn,
      images: turn.images?.map((img) => ({ bytes: img.bytes.slice(0), mime: img.mime })),
    }));

    const imagePayloads = currentImages.map((bytes) => ({
      bytes: bytes.slice(0),
      mime: currentImageMime,
    }));

    const transferables: ArrayBuffer[] = [];
    for (const p of imagePayloads) transferables.push(p.bytes);
    for (const turn of historyForWorker) {
      if (turn.images) {
        for (const img of turn.images) transferables.push(img.bytes);
      }
    }

    workerRef.current?.postMessage(
      {
        type: "generate",
        modelId: GEMMA_MODEL_ID,
        prompt: promptText,
        history: historyForWorker,
        images: imagePayloads,
        systemPrompt,
        maxNewTokens,
        temperature,
        repetitionPenalty,
        topP,
        thinkingMode: thinkingRef.current,
        webContext,
      },
      transferables,
    );

    pushActivity({
      direction: "out",
      kind: "generate",
      title: "יצירת תשובה (Chat)",
      detail: `SYSTEM:\n${systemPrompt}\n\nUSER:\n${promptText}${webContext ? `\n\nWEB CONTEXT:\n${webContext}` : ""}`,
      meta: {
        maxNewTokens,
        temperature,
        topP,
        repetitionPenalty,
        thinking: thinkingRef.current,
        images: imagePayloads.length,
        historyTurns: history.length,
      },
    });
  };

  useEffect(() => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./model.worker.ts", import.meta.url), {
        type: "module",
      });
    } catch (e) {
      console.error("[GROVEE] Worker constructor failed:", e);
      queueMicrotask(() => {
        setWorkerBootError(e instanceof Error ? e.message : String(e));
      });
      return () => {};
    }

    worker.onerror = (ev: ErrorEvent) => {
      console.error("[GROVEE] worker script error:", ev.message, ev.filename, ev.lineno);
      setWorkerBootError(
        ev.message || `Worker script failed (404/CORS?). File: ${ev.filename ?? "model.worker"}`,
      );
      setIsLoading(false);
    };

    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;
      lastChatSignalRef.current = Date.now();
      if (msg.type === "status") {
        setStatus(msg.text);
      } else if (msg.type === "progress") {
        setStatus(msg.text);
        if (isLoadingRef.current) {
          if (msg.phase === "init") {
            setProgress(msg.progress);
            loadingFileRef.current = "";
            setLoadingFile("");
            setLoadingBytes({ loaded: 0, total: 0, speedBps: 0 });
          } else {
            const nextFile = msg.file ?? "";
            const nextLoaded = typeof msg.loaded === "number" ? msg.loaded : 0;
            const nextTotal = typeof msg.total === "number" ? msg.total : 0;
            const nextSpeed = typeof msg.speedBps === "number" ? msg.speedBps : 0;

            if (nextFile) {
              loadingFileRef.current = nextFile;
              setLoadingFile(nextFile);
            }
            const bytePct =
              nextTotal > 0 && nextLoaded > 0
                ? downloadProgressPercent(nextLoaded, nextTotal)
                : 0;
            if (bytePct > 0) {
              setProgress(bytePct);
            } else {
              setProgress((prev) => Math.max(prev, msg.progress));
            }
            setLoadingBytes((prev) => ({
              loaded: Math.max(prev.loaded, nextLoaded),
              total: Math.max(prev.total, nextTotal),
              speedBps: nextSpeed > 0 ? nextSpeed : prev.speedBps,
            }));
          }
          setLoadingPhase(msg.phase === "init" ? "init" : "download");
        }
      } else if (msg.type === "loaded") {
        setWorkerBootError(null);
        setIsGemmaLoaded(true);
        setChatModelAvailability("gemma");
        setCapabilitiesFailureReason(null);
        setIsLoaded(true);
        setIsLoading(false);
        setProgress(100);
        setStatus(`Gemma ready on ${formatInferenceDevice(msg.device)}`);
      } else if (msg.type === "token") {
        if (globeHeadlineModeRef.current) {
          globeHeadlineBufferRef.current += msg.text;
          return;
        }
        streamTokenCountRef.current += 1;
        setStreamTokenCount(streamTokenCountRef.current);
        setAssistantBuffer((prev) => {
          const next = prev + msg.text;
          assistantBufferRef.current = next;
          return next;
        });
      } else if (msg.type === "done") {
        if (globeHeadlineModeRef.current) {
          const parsed = parseHeadlineLines(globeHeadlineBufferRef.current);
          globeHeadlineModeRef.current = false;
          globeHeadlineBufferRef.current = "";
          publishGlobeHeadlineResult(parsed);
          return;
        }
        pushActivity({
          direction: "in",
          kind: "generate",
          title: "תשובת צ'אט הושלמה",
          detail: assistantBufferRef.current || "(empty)",
          meta: { chars: assistantBufferRef.current.length },
        });
        finalizeAssistantReply(false);
      } else if (msg.type === "aborted") {
        pushActivity({
          direction: "in",
          kind: "generate",
          title: "יצירה נעצרה",
          detail: assistantBufferRef.current || "(empty)",
        });
        finalizeAssistantReply(true);
      } else if (msg.type === "error") {
        if (globeHeadlineModeRef.current) {
          globeHeadlineModeRef.current = false;
          globeHeadlineBufferRef.current = "";
          publishGlobeHeadlineResult([]);
          return;
        }
        pushActivity({
          direction: "in",
          kind: "error",
          title: "שגיאת worker",
          detail: msg.error,
        });
        const isChatError = msg.scope === "chat" || isGeneratingRef.current;
        if (isChatError) {
          isGeneratingRef.current = false;
          setIsGenerating(false);
          syncVisionBusy();
          const errText = msg.error.trim();
          const recovery =
            /unaligned|alignment/i.test(errText)
              ? "השיחה ארוכה מדי לעיבוד — נסה צ'אט חדש או שאלה קצרה יותר."
              : errText;
          setAssistantBuffer((prev) => {
            const next = prev.trim()
              ? `${prev.trim()}\n\n⚠️ ${recovery}`
              : `⚠️ ${recovery}`;
            assistantBufferRef.current = next;
            return next;
          });
          finalizeAssistantReply(true);
          setStatus(`שגיאה: ${errText.slice(0, 120)}`);
          cameraLoopRef.current?.releaseAfterChat();
        } else {
          const errText = msg.error;
          if (isWebGpuInferenceError(errText) && !wasmBootRetryRef.current) {
            wasmBootRetryRef.current = true;
            setWorkerBootError(null);
            setStatus("WebGPU לא זמין — ממשיך ב-WASM (CPU)…");
            loadModel({ forceWasm: true });
            return;
          }
          if (isLoadingRef.current) {
            enterCapabilitiesOnlyModeRef.current(errText);
            return;
          }
          if (isWebGpuInferenceError(errText)) {
            try {
              localStorage.setItem(WEBGPU_BLOCKED_KEY, "1");
            } catch {
              /* ignore */
            }
          }
          setIsLoading(false);
          setProgress(0);
          setStatus(`Error: ${errText}`);
          setWorkerBootError(
            isWebGpuInferenceError(errText)
              ? `${errText} — לחץ «טען ב-WASM» למטה, או הגדרות → Inference → WASM, נקה מטמון, והתחל שוב.`
              : errText,
          );
        }
      } else if (msg.type === "scene_analysis") {
        workerInferenceBusyRef.current = false;
        const meta = sceneAnalysisMetaRef.current.get(msg.requestId);
        sceneAnalysisMetaRef.current.delete(msg.requestId);
        const resolve = sceneAnalysisResolversRef.current.get(msg.requestId);
        if (resolve) {
          sceneAnalysisResolversRef.current.delete(msg.requestId);
          if (msg.ok && msg.summary !== undefined) {
            pushActivity({
              direction: "in",
              kind: "analyze_scene",
              title: "תוצאת ניתוח סצנה",
              detail:
                msg.raw ??
                JSON.stringify(
                  {
                    summary: msg.summary,
                    current: msg.current,
                    events: msg.events,
                    interesting: msg.interesting,
                  },
                  null,
                  2,
                ),
              meta: {
                interesting: msg.interesting ?? false,
                events: msg.events?.length ?? 0,
                reason: meta?.reason ?? "?",
              },
            });
            resolve({
              objects: msg.objects ?? msg.current ?? [],
              people: msg.people ?? [],
              current: msg.current ?? msg.objects ?? [],
              events: msg.events ?? [],
              interesting: msg.interesting ?? false,
              summary: msg.summary ?? "",
            });
          } else {
            const skipped = msg.error === "chat_active" || msg.error === "scene_busy" || msg.error === "busy";
            if (!skipped) {
              pushActivity({
                direction: "in",
                kind: "analyze_scene",
                title: "ניתוח סצנה נכשל",
                detail: msg.error ?? "unknown error",
                meta: { reason: meta?.reason ?? "?" },
              });
            }
            resolve(null);
          }
        }
      } else if (msg.type === "character_utterance_token") {
        const listener = characterUtteranceTokenListenersRef.current.get(msg.requestId);
        listener?.onCount?.(msg.tokens);
        if (msg.text) listener?.onChunk?.(msg.text);
      } else if (msg.type === "character_utterance") {
        workerInferenceBusyRef.current = false;
        const resolveUtterance = characterUtteranceResolversRef.current.get(msg.requestId);
        if (resolveUtterance) {
          characterUtteranceResolversRef.current.delete(msg.requestId);
          characterUtteranceTokenListenersRef.current.delete(msg.requestId);
          if (msg.ok && msg.text?.trim()) {
            pushActivity({
              direction: "in",
              kind: "character_utterance",
              title: "דיבור יזום (Gemma)",
              detail: msg.text.trim(),
              meta: { chars: msg.text.trim().length },
            });
            resolveUtterance(msg.text.trim());
          } else {
            resolveUtterance(null);
          }
        }
      } else if (msg.type === "search_plan") {
        workerInferenceBusyRef.current = false;
        const resolvePlan = searchPlanResolversRef.current.get(msg.requestId);
        if (resolvePlan) {
          searchPlanResolversRef.current.delete(msg.requestId);
          if (msg.ok && msg.text?.trim()) {
            pushActivity({
              direction: "in",
              kind: "web_search",
              title: "Search plan (Gemma)",
              detail: msg.text.trim().slice(0, 1200),
            });
            resolvePlan(msg.text.trim());
          } else {
            resolvePlan(null);
          }
        }
      }
    };

    workerRef.current = worker;
    worker.postMessage({
      type: "configure_hub",
      remoteHost: appSettingsRef.current.hfRemoteHost ?? "",
    });
    worker.postMessage({
      type: "configure_inference",
      backend: appSettingsRef.current.inferenceBackend,
    });
    void (async () => {
      const r = await requestPersistentStorage();
      console.info("[GROVEE] storage.persist:", r);
    })();
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [finalizeAssistantReply, setMessages, workerReloadKey, pushActivity, syncVisionBusy]);

  useEffect(() => {
    workerRef.current?.postMessage({
      type: "configure_hub",
      remoteHost: appSettings.hfRemoteHost ?? "",
    });
  }, [appSettings.hfRemoteHost]);

  useEffect(() => {
    workerRef.current?.postMessage({
      type: "configure_inference",
      backend: appSettings.inferenceBackend,
    });
  }, [appSettings.inferenceBackend]);

  useEffect(() => {
    return subscribeGlobeHeadlineRequests((ctx) => {
      if (!workerRef.current || !isLoaded || isGenerating || globeHeadlineModeRef.current) {
        publishGlobeHeadlineResult([]);
        return;
      }
      globeHeadlineModeRef.current = true;
      globeHeadlineBufferRef.current = "";
      workerRef.current.postMessage({
        type: "generate",
        modelId: GEMMA_MODEL_ID,
        prompt: buildGlobeHeadlinePrompt(ctx),
        history: [],
        images: [],
        systemPrompt: GLOBE_HEADLINE_SYSTEM,
        maxNewTokens: 120,
        temperature: 0.55,
        repetitionPenalty: 1.08,
        topP: 0.9,
        thinkingMode: false,
        webContext: "",
      });
    });
  }, [isLoaded, isGenerating]);

  const enterCapabilitiesOnlyModeRef = useRef<(failureReason?: string) => void>(() => {});

  const enterCapabilitiesOnlyMode = useCallback((failureReason?: string) => {
    setChatModelAvailability("none");
    setCapabilitiesFailureReason(failureReason?.trim() || null);
    setWorkerBootError(null);
    setIsLoading(false);
    setIsGemmaLoaded(false);
    setProgress(0);
    setLocalTextDownloadingId(null);
    setLocalTextDownloadPct(0);
    setLocalTextDownloadLabel("");
    terminateLocalTextWorker();
    const rack = loadModelRack();
    setModelRack(rack);
    const defaultId = pickCapabilitiesDefaultRackId(rack);
    if (defaultId) {
      setSelectedRackModelId(defaultId);
      persistSelectedModelId(defaultId);
    }
    setIsLoaded(true);
    setStatus("מצב יכולות — אין מודל שיחה");
  }, []);

  useEffect(() => {
    enterCapabilitiesOnlyModeRef.current = enterCapabilitiesOnlyMode;
  }, [enterCapabilitiesOnlyMode]);

  const loadLocalTextBoot = async () => {
    setWorkerBootError(null);
    setIsLoading(true);
    setIsLoaded(false);
    setBootTarget("local-text");
    setStatus("טוען SmolLM2…");
    setProgress(0);
    setLoadingPhase("download");
    setLoadingBytes({ loaded: 0, total: 0, speedBps: 0 });
    setLoadingTipIndex(0);
    loadingFileRef.current = "";
    setLoadingFile("");
    setSelectedRackModelId(SMOLLM_RACK_ID);
    persistSelectedModelId(SMOLLM_RACK_ID);

    const lt = appSettingsRef.current.localText;
    const alreadyReady = readLocalTextReadyIds().includes(SMOLLM_RACK_ID);

    try {
      if (!alreadyReady) {
        await downloadLocalTextModel(
          SMOLLM_RACK_ID,
          SMOLLM_HF_MODEL_ID,
          (p) => {
            setLocalTextDownloadPct(p.pct);
            setLocalTextDownloadLabel(p.message);
            setProgress(p.pct);
            if (p.message) {
              loadingFileRef.current = p.message;
              setLoadingFile(p.message);
            }
            setLoadingBytes({
              loaded: p.loaded,
              total: p.total,
              speedBps: 0,
            });
          },
          lt.inferenceBackend,
        );
      } else {
        setProgress(100);
        setStatus("SmolLM כבר מותקן — מכין…");
      }
      setModelRack(loadModelRack());
      setWorkerBootError(null);
      setChatModelAvailability("local-text");
      setCapabilitiesFailureReason(null);
      setIsLoaded(true);
      setIsLoading(false);
      setProgress(100);
      setStatus("SmolLM מוכן לשיחה");
      setLocalTextDownloadingId(null);
      setLocalTextDownloadPct(0);
      setLocalTextDownloadLabel("");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      enterCapabilitiesOnlyMode(msg);
    }
  };

  const startIntroLoad = () => {
    void (async () => {
      const rec = await resolveStartupModelChoice(appSettingsRef.current.startupModel);
      setStartupRecommendation(rec);
      setBootTarget(rec.choice);
      if (rec.choice === "local-text") {
        await loadLocalTextBoot();
      } else {
        loadModel();
      }
    })();
  };

  const loadModel = (opts?: { forceWasm?: boolean; persistWasm?: boolean }) => {
    if (!workerRef.current) return;
    setBootTarget("gemma");
    if (!opts?.forceWasm) wasmBootRetryRef.current = false;
    setWorkerBootError(null);
    setIsLoading(true);
    setIsLoaded(false);
    setStatus(opts?.forceWasm ? "טוען מודל ב-WASM (CPU)…" : "Loading Gemma 4 E2B…");
    setProgress(0);
    setLoadingPhase("download");
    setLoadingBytes({ loaded: 0, total: 0, speedBps: 0 });
    setLoadingTipIndex(0);
    loadingFileRef.current = "";
    setLoadingFile("");
    const backend = opts?.forceWasm ? "wasm" : appSettingsRef.current.inferenceBackend;
    if (opts?.forceWasm && opts?.persistWasm && appSettingsRef.current.inferenceBackend !== "wasm") {
      const next = { ...appSettingsRef.current, inferenceBackend: "wasm" as const };
      appSettingsRef.current = next;
      setAppSettings(next);
      saveSettings(next);
      try {
        localStorage.setItem(WEBGPU_BLOCKED_KEY, "1");
      } catch {
        /* ignore */
      }
    }
    workerRef.current.postMessage({
      type: "configure_hub",
      remoteHost: appSettingsRef.current.hfRemoteHost ?? "",
    });
    workerRef.current.postMessage({
      type: "load",
      modelId: GEMMA_MODEL_ID,
      dtype: "q4",
      backend,
    });
  };

  const retryWasmLoad = () => loadModel({ forceWasm: true, persistWasm: true });

  const clearModelCache = async () => {
    if (isGenerating || cacheClearing) return;
    setCacheClearing(true);

    const estimateUsage = async (): Promise<number> => {
      try {
        if (!navigator.storage?.estimate) return -1;
        const est = await navigator.storage.estimate();
        return typeof est.usage === "number" ? est.usage : -1;
      } catch {
        return -1;
      }
    };

    setStatus("מנקה מטמון…");
    const before = await estimateUsage();

    try {
      workerRef.current?.postMessage({ type: "clear_runtime_cache" });
      try {
        workerRef.current?.terminate();
      } catch {
        /* noop */
      }
      workerRef.current = null;

      if ("caches" in self) {
        try {
          const keys = await caches.keys();
          for (const k of keys) {
            await caches.delete(k);
          }
        } catch {
          /* ignore */
        }
      }

      const idb = indexedDB as IDBFactory & {
        databases?: () => Promise<Array<{ name?: string }>>;
      };
      if (idb.databases) {
        try {
          const dbs = await idb.databases();
          for (const db of dbs) {
            if (db.name) {
              await new Promise<void>((resolve) => {
                const req = indexedDB.deleteDatabase(db.name!);
                req.onsuccess = () => resolve();
                req.onerror = () => resolve();
                req.onblocked = () => resolve();
              });
            }
          }
        } catch {
          /* ignore */
        }
      }

      const storageWithDir = navigator.storage as StorageManager & {
        getDirectory?: () => Promise<FileSystemDirectoryHandle>;
      };
      if (storageWithDir.getDirectory) {
        try {
          const root = await storageWithDir.getDirectory();
          const walker = root as FileSystemDirectoryHandle & {
            entries?: () => AsyncIterableIterator<[string, FileSystemHandle]>;
            keys?: () => AsyncIterableIterator<string>;
            removeEntry: (name: string, opts?: { recursive?: boolean }) => Promise<void>;
          };
          const collected: string[] = [];
          if (walker.entries) {
            for await (const [name] of walker.entries()) collected.push(name);
          } else if (walker.keys) {
            for await (const name of walker.keys()) collected.push(name);
          }
          for (const name of collected) {
            try {
              await walker.removeEntry(name, { recursive: true });
            } catch {
              /* ignore */
            }
          }
        } catch {
          /* ignore */
        }
      }

      const after = await estimateUsage();
      setProgress(0);
      setIsLoaded(false);
      setIsGemmaLoaded(false);
      setChatModelAvailability("gemma");
      setCapabilitiesFailureReason(null);
      setIsLoading(false);
      setAssistantBuffer("");
      assistantBufferRef.current = "";
      setWorkerReloadKey((k) => k + 1);

      try {
        localStorage.removeItem(WEBGPU_BLOCKED_KEY);
      } catch {
        /* ignore */
      }
      clearStartupContextCache();

      const freedSummary =
        before >= 0 && after >= 0
          ? `שוחררו ${formatBytes(Math.max(0, before - after))} · נותר: ${formatBytes(after)}`
          : "ניקוי הסתיים";
      setStatus(`${freedSummary}. לחץ «התחל» לטעינה מחדש.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`ניקוי מטמון נכשל: ${message}`);
    } finally {
      setCacheClearing(false);
    }
  };

  const persistSettings = (s: AppSettings) => {
    const normalized: AppSettings = {
      ...s,
      inferenceBackend: normalizeInferenceBackend(s.inferenceBackend),
      localText: mergeLocalTextSettings(s.localText),
    };
    setAppSettings((prev) => {
      if (
        normalized.inferenceBackend !== prev.inferenceBackend ||
        normalized.hfRemoteHost !== prev.hfRemoteHost
      ) {
        queueMicrotask(() => {
          setIsLoaded(false);
          setIsLoading(false);
          setStatus("הגדרות השתנו — לחץ «התחל» כדי לטעון מחדש");
        });
      }
      if (JSON.stringify(normalized.localText) !== JSON.stringify(prev.localText)) {
        terminateLocalTextWorker();
      }
      return normalized;
    });
    saveSettings(normalized);
  };

  type BeginGenerationOptions = {
    trimmed: string;
    effectivePrompt: string;
    priorMessages: ChatMessage[];
    attachmentSnapshot: PendingAttachment[];
    restoredImageBuffers?: ArrayBuffer[];
  };

  const beginGeneration = async ({
    trimmed,
    effectivePrompt,
    priorMessages,
    attachmentSnapshot,
    restoredImageBuffers = [],
  }: BeginGenerationOptions) => {
    const cameraActive = cameraModeRef.current;
    const hasAttachmentsEarly = attachmentSnapshot.length > 0 || restoredImageBuffers.length > 0;
    const documentTurn = needsAttachedDocumentAnalysis(trimmed, hasAttachmentsEarly);
    generationChatOnlyDocumentRef.current = documentTurn;
    setChatOnlyDocumentMode(documentTurn);
    generationCameraModeRef.current = cameraActive;
    const prevChatTopic = chatTopicRef.current;
    const chatTopic = classifyChatTopic(trimmed);
    const topicShifted = isTopicShift(prevChatTopic, chatTopic);
    chatTopicRef.current = chatTopic;

    const priorTurns = cameraActive
      ? buildCameraHistoryForWorker(cameraStoreRef.current.messages)
      : buildHistoryForWorker(priorMessages);

    const continueCode = shouldContinueCode(effectivePrompt, priorTurns);
    continueModeRef.current = continueCode;

    const hasAttachments = attachmentSnapshot.length > 0 || restoredImageBuffers.length > 0;

    if (continueCode) {
      const lastAssistant = [...priorTurns].reverse().find((t) => t.role === "assistant");
      const seed = lastAssistant?.content ?? "";
      setAssistantBuffer(seed);
      assistantBufferRef.current = seed;
      setArtifactOpen(true);
      if (lastAssistant) {
        const art = extractPrimaryArtifact(seed);
        if (art) setActiveArtifact(art);
      }
    } else {
      setAssistantBuffer("");
      assistantBufferRef.current = "";
    }
    streamTokenCountRef.current = 0;
    setStreamTokenCount(0);

    isGeneratingRef.current = true;
    setIsGenerating(true);
    syncVisionBusy();

    const g = appSettings.gemma;
    const greeting = isSimpleGreeting(effectivePrompt) && !hasAttachments;
    const fingerCountQuestion = isFingerCountQuestion(trimmed);
    const hasWorldData = worldMemoryRef.current.hasData();
    const greetingWithCamera = greeting && cameraActive && hasWorldData;
    const greetingCameraStarting = greeting && cameraActive && !hasWorldData;
    const visualDetailQuestion = isVisualDetailQuestion(trimmed);
    const personVisibilityQuestion = isPersonVisibilityQuestion(trimmed);
    const personActivityQuestion = isPersonActivityQuestion(trimmed);
    const personStateQuestion = isCurrentPersonStateQuestion(trimmed);
    const personFocusRefresh = needsPersonFocusRefresh(trimmed);
    const sceneInterpretation = isSceneInterpretationQuestion(trimmed);
    const consciousnessQuestion = isConsciousnessQuestion(trimmed);
    const personDemographicsQuestion = isPersonDemographicsQuestion(trimmed);
    const personMoodQuestion = isPersonMoodQuestion(trimmed);
    const conversationFirst = isConversationFirstRequest(trimmed);
    const visionSensorQuery =
      cameraActive && needsVisionSensorContext(trimmed) && !documentTurn;
    const pureChatTurn = cameraActive && isVisionUnrelatedTurn(trimmed) && !documentTurn;
    lastTurnUsedVisionRef.current = visionSensorQuery;
    const liveCameraContext = needsLiveCameraContext(trimmed);
    let webContext = "";
    let searchHint = "";
    let marineLiveCannedReply: string | null = null;
    pendingWebSearchRef.current = null;
    pendingTimeWidgetRef.current = null;
    const wantsGameSearch =
      !cameraActive &&
      !hasAttachments &&
      shouldOpenGamePanel(trimmed || effectivePrompt, chatTopic);
    const localTimeOnly =
      !hasAttachments &&
      !wantsGameSearch &&
      startupContext &&
      isLocalContextTimeQuery(trimmed || effectivePrompt);
    const shouldRunWebSearch =
      !hasAttachments &&
      !wantsGameSearch &&
      !localTimeOnly &&
      needsWebSearch(trimmed || effectivePrompt);
    let searchIntentsForGlobe: SearchIntent[] = [];
    let lastSearchSources: SearchSourceResult[] = [];

    const deliverLiveCannedReply = (reply: string, ctx: string, title = "Live data · canned reply"): boolean => {
      const text = reply.trim();
      if (!text) return false;
      if (qaTurnForceLlmRef.current || qaChatBridge.isForceLlmPending()) return false;
      if (cameraActive || hasAttachments || continueCode || documentTurn || wantsGameSearch) return false;
      qaChatBridge.setWebContext(ctx);
      qaChatBridge.setReplySource("canned-live");
      assistantBufferRef.current = text;
      setAssistantBuffer(text);
      setStatus("Ready");
      pushActivity({
        direction: "system",
        kind: "web_search",
        title,
        detail: text.slice(0, 1200),
      });
      finalizeAssistantReply(false);
      return true;
    };

    const finishCannedLive = (reply: string, ctx: string): boolean => {
      if (!qaChatBridge.hasPending()) return false;
      return deliverLiveCannedReply(reply, ctx);
    };

    if (!shouldRunWebSearch && !localTimeOnly) {
      const preCanned = buildCapabilityLiveReply(effectivePrompt, [], []);
      if (preCanned && isStarlinkRegionalQuery(effectivePrompt)) {
        if (finishCannedLive(preCanned, "")) return;
      }
    }

    if (localTimeOnly && startupContext) {
      pendingTimeWidgetRef.current = buildTimeWidgetFromStartupContext(startupContext);
      webContext = buildLocalTimeAnswer(startupContext, effectivePrompt);
      searchHint = " · זמן מקומי (ללא חיפוש ברשת)";
      pushActivity({
        direction: "system",
        kind: "web_search",
        title: "Local Context",
        detail: webContext,
      });
    } else if (shouldRunWebSearch) {
      setStatus("מחפש מידע…");
      setStreamingSearchSources({
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
        const searchPlan = await resolveSearchPlanForQuery(effectivePrompt, recentUserText);
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
              setStreamingSearchSources((prev) => {
                if (!prev) return prev;
                const rest = prev.sources.filter((s) => s.provider !== ev.result.provider);
                return { ...prev, sources: [...rest, ev.result] };
              });
            }
            if (ev.type === "complete") {
              setStreamingSearchSources((prev) =>
                prev
                  ? {
                      ...prev,
                      sources: ev.sources,
                      active: false,
                    }
                  : null,
              );
            }
          },
        });
        searchIntentsForGlobe = searchResult.intents;
        lastSearchSources = searchResult.sources;
        webContext = searchResult.contextText;
        const unifiedSearchPayload = buildUnifiedSearchPayload(effectivePrompt, searchResult.sources);
        const placeOrRouteOnly =
          (searchResult.intents.includes("places") || searchResult.intents.includes("distance")) &&
          shouldOpenGlobeForStructuredGeo(effectivePrompt, searchResult.intents, searchResult.sources) &&
          !unifiedSearchPayload.hits.some((h) =>
            ["rss", "web", "youtube", "video", "image", "movie"].includes(h.kind),
          );
        if (shouldOpenSearchResultsPanel(unifiedSearchPayload) && !placeOrRouteOnly) {
          setSearchResultsPayload(unifiedSearchPayload);
          setSearchResultsOpen(true);
          setArtifactOpen(false);
          setGlobePanelOpen(false);
          setGamesPanelOpen(false);
        }
        const newsPayloadAfterSearch = unifiedSearchPayload.hits.length
          ? { cardCount: unifiedSearchPayload.facets.rss, mode: "search" as const }
          : null;
        if (isSinglePlaceTimeWidgetQuery(effectivePrompt)) {
          const wt = searchResult.sources.find((s) => s.provider === "world-time" && s.ok);
          const widget = wt ? buildTimeWidgetFromWorldTimeSource(wt) : null;
          if (widget) pendingTimeWidgetRef.current = widget;
        }
        const searchLiveOk = searchResult.sources.some((s) => s.ok && s.text.trim());
        marineLiveCannedReply = searchResult.cannedReply ?? buildCapabilityLiveReply(
          effectivePrompt,
          searchResult.intents,
          searchResult.sources,
          {
            answerShape: searchPlan?.answerShape,
          },
        );
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
            {
              answerShape: searchPlan?.answerShape,
            },
          );
        }
        const newsQueryTurn = isNewsQuery(effectivePrompt);
        const newsHeadlineBulletsTurn = wantsNewsHeadlineBulletsInChat(effectivePrompt);
        const qaForceLlm = qaTurnForceLlmRef.current || qaChatBridge.isForceLlmPending();
        const shouldDeliverLive =
          !qaForceLlm &&
          !!marineLiveCannedReply &&
          !cameraActive &&
          !hasAttachments &&
          !continueCode &&
          !documentTurn &&
          !wantsGameSearch &&
          (!newsQueryTurn || newsHeadlineBulletsTurn) &&
          (searchLiveOk || !searchResult.sources.some((s) => s.ok && s.provider !== "searxng")) &&
          shouldDeliverStructuredLiveReply(
            effectivePrompt,
            searchResult.intents,
            searchResult.sources,
            marineLiveCannedReply,
          );
        // #region agent log
        agentDebugLog("H4,H5", "App.tsx:webSearchDecision", "web search result delivery decision", {
          queryPreview: effectivePrompt.slice(0, 120),
          intents: searchResult.intents,
          sourceLabels: searchResult.sources.map((s) => ({ provider: s.provider, label: s.label, ok: s.ok, hasText: !!s.text.trim(), error: s.error?.slice(0, 120) })),
          searchLiveOk,
          cannedReplyExists: !!marineLiveCannedReply,
          shouldDeliverLive,
          blockingFlags: { forceLlm: qaForceLlm, cameraActive, hasAttachments, continueCode, documentTurn, wantsGameSearch },
          pendingTimeWidget: !!pendingTimeWidgetRef.current,
        });
        // #endregion
        if (
          !searchLiveOk &&
          !marineLiveCannedReply &&
          (searchPlan?.useWebFallback || searchResult.sources.some((s) => s.provider === "searxng"))
        ) {
          marineLiveCannedReply = buildWebFallbackNoDataReply(effectivePrompt, searchResult.sources);
        }
        pendingWebSearchRef.current = {
          sources: searchResult.sources,
          summary: searchResult.summaryHe,
          query: effectivePrompt,
          answerShape: searchPlan?.answerShape,
          crossSource:
            isCrossSourceQuery(effectivePrompt) ||
            searchResult.intents.length >= 2 ||
            !!searchPlan?.blendNewsWithWeb,
        };
        setStreamingSearchSources({
          sources: searchResult.sources,
          summary: searchResult.summaryHe,
          query: effectivePrompt,
          brief: searchResult.brief,
          active: false,
        });
        if (
          shouldDeliverLive &&
          marineLiveCannedReply &&
          deliverLiveCannedReply(marineLiveCannedReply, webContext, "Live data · canned reply")
        ) {
          return;
        }
        if (
          needsOpenWebEnrichment(effectivePrompt) &&
          !wantsCinemaPlotSummaries(effectivePrompt) &&
          marineLiveCannedReply &&
          !cameraActive &&
          !hasAttachments &&
          !continueCode &&
          !documentTurn &&
          !wantsGameSearch &&
          !qaTurnForceLlmRef.current &&
          !qaChatBridge.isForceLlmPending() &&
          deliverLiveCannedReply(marineLiveCannedReply, webContext, "Open web · canned reply")
        ) {
          return;
        }
        if (
          newsQueryTurn &&
          !newsHeadlineBulletsTurn &&
          !cameraActive &&
          !hasAttachments &&
          !continueCode &&
          !documentTurn &&
          !wantsGameSearch &&
          !qaTurnForceLlmRef.current &&
          !qaChatBridge.isForceLlmPending()
        ) {
          const guide = buildNewsPanelGuideReply(
            effectivePrompt,
            newsPayloadAfterSearch
              ? {
                  mode: newsPayloadAfterSearch.mode,
                  cardCount: newsPayloadAfterSearch.cardCount,
                }
              : null,
          );
          if (deliverLiveCannedReply(guide, "", "GROVEE NEWS · פאנל")) {
            return;
          }
        }
        if (!webContext.trim()) {
          searchHint = " · אין תוצאות חיפוש";
        } else {
          searchHint = unifiedSearchPayload.hits.length
            ? ` · ${unifiedSearchPayload.hits.length} תוצאות בפאנל · ${searchResult.summaryHe}`
            : ` · ${searchResult.summaryHe}`;
        }
        pushActivity({
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
        pendingWebSearchRef.current = null;
        setStreamingSearchSources(null);
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
      setGlobePanelOpen(false);
      setGlobeCommand(null);
    }
    if (openGlobe) {
      const cmd =
        globeFromSearch ?? buildGlobeCommand(trimmed || effectivePrompt, searchIntentsForGlobe);
      if (cmd) {
        setGlobePanelOpen(true);
        setGlobeCommand(cmd);
        setArtifactOpen(false);
        setGamesPanelOpen(false);
        setSearchResultsOpen(false);
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
    pendingGameCategoryPickerRef.current = false;
    pendingGameBrowseCategoryRef.current = null;
    setStreamingGameCategoryPicker(false);
    let gameGroundingBlock = "";
    let gameNoResults = false;
    let gameSearchCannedReply: string | null = null;
    if (wantsGameSearch) {
      setStreamingSearchSources(null);
      setStatus("מחפש משחקים…");
      try {
        const gameReq = parseGameUserRequest(trimmed || effectivePrompt);
        const panelCategory = gameReq.category ?? "featured";
        const gameResult = await searchOnlineGamesWithFallback(gameReq, 12);
        setGamesPanelCategory(panelCategory);
        setGamesPanelLayout("side");
        setGamesPanelOpen(true);
        setArtifactOpen(false);
        setGamesEmbedGame(null);

        if (gameResult.matchFound && gameResult.games.length) {
          setGamesPanelGames(gameResult.games);
          setGamesPanelTitle(gameReq.panelTitle);
          gameSearchHint = ` · ${gameResult.games.length} משחקים`;
          gameGroundingBlock = gameResult.games.map((g, i) => `${i + 1}. ${g.title}`).join("\n");
          gameSearchCannedReply = buildGameSearchFoundReply(gameResult.games.length, gameReq);
          pushActivity({
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
          setGamesPanelGames([]);
          setGamesPanelTitle(gameReq.panelTitle);
          pendingGameCategoryPickerRef.current = true;
          pendingGameBrowseCategoryRef.current = panelCategory;
          setStreamingGameCategoryPicker(true);
          gameSearchHint = " · לא נמצא — קטגוריות";
          gameSearchCannedReply = buildGameSearchNotFoundReply(gameReq);
          pushActivity({
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
        pendingGameCategoryPickerRef.current = true;
        setStreamingGameCategoryPicker(true);
        setGamesPanelLayout("side");
        setGamesPanelOpen(true);
        setGamesPanelGames([]);
        gameSearchCannedReply = buildGameSearchNotFoundReply(
          parseGameUserRequest(trimmed || effectivePrompt),
        );
      }
    }

    const qaForceLlm = qaTurnForceLlmRef.current || qaChatBridge.isForceLlmPending();

    const pureGameSearchTurn =
      !qaForceLlm && wantsGameSearch && !cameraActive && !hasAttachments && !continueCode && !documentTurn;
    if (pureGameSearchTurn && gameSearchCannedReply) {
      qaChatBridge.setReplySource("canned-game");
      assistantBufferRef.current = gameSearchCannedReply;
      setAssistantBuffer(gameSearchCannedReply);
      setStatus("Ready");
      finalizeAssistantReply(false);
      return;
    }

    const pureGlobePlaceTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      !!globePlaceCannedReply;
    if (pureGlobePlaceTurn && globePlaceCannedReply) {
      qaChatBridge.setReplySource("canned-globe");
      assistantBufferRef.current = globePlaceCannedReply;
      setAssistantBuffer(globePlaceCannedReply);
      setStatus("Ready");
      pushActivity({
        direction: "system",
        kind: "globe_focus",
        title: "Globe · place focus",
        detail: globePlaceLabel || globePlaceCannedReply,
      });
      finalizeAssistantReply(false);
      return;
    }

    const pureTimeWidgetTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      !!pendingTimeWidgetRef.current &&
      isSinglePlaceTimeWidgetQuery(effectivePrompt);
    // #region agent log
    agentDebugLog("H5", "App.tsx:pureTimeWidgetTurn", "time widget direct reply decision", {
      queryPreview: effectivePrompt.slice(0, 120),
      pureTimeWidgetTurn,
      pendingTimeWidget: !!pendingTimeWidgetRef.current,
      qaForceLlm,
      blockingFlags: { wantsGameSearch, cameraActive, hasAttachments, continueCode, documentTurn },
    });
    // #endregion
    if (pureTimeWidgetTurn && pendingTimeWidgetRef.current) {
      const reply = buildShortTimeReply(pendingTimeWidgetRef.current);
      qaChatBridge.setReplySource("local-time");
      assistantBufferRef.current = reply;
      setAssistantBuffer(reply);
      setStatus("Ready");
      pushActivity({
        direction: "system",
        kind: "web_search",
        title: "Time widget",
        detail: reply,
      });
      finalizeAssistantReply(false);
      return;
    }

    const structuredLiveTurn =
      !qaForceLlm &&
      !isNewsQuery(effectivePrompt) &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      !!marineLiveCannedReply &&
      shouldDeliverStructuredLiveReply(
        effectivePrompt,
        searchIntentsForGlobe,
        lastSearchSources,
        marineLiveCannedReply,
      );
    if (structuredLiveTurn && marineLiveCannedReply && !globePlaceCannedReply) {
      if (deliverLiveCannedReply(marineLiveCannedReply, webContext)) return;
    }

    const pureCurrencyTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      isCurrencyQuery(effectivePrompt) &&
      !!marineLiveCannedReply;
    if (pureCurrencyTurn && marineLiveCannedReply) {
      if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "Frankfurter · canned reply")) return;
    }

    const pureEarthquakeTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      isEarthquakeQuery(effectivePrompt) &&
      lastSearchSources.some((s) => s.provider === "usgs-earthquake" && s.ok && s.text.trim()) &&
      !!marineLiveCannedReply;
    if (pureEarthquakeTurn && marineLiveCannedReply) {
      if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "USGS · canned reply")) return;
    }

    const pureDisasterTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      isDisasterQuery(effectivePrompt) &&
      !isEarthquakeQuery(effectivePrompt) &&
      lastSearchSources.some((s) => s.provider === "gdacs-disasters" && s.ok && s.text.trim()) &&
      !!marineLiveCannedReply;
    if (pureDisasterTurn && marineLiveCannedReply) {
      if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "GDACS · canned reply")) return;
    }

    const pureAviationTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      (isAviationQuery(effectivePrompt) || /כמה\s+מטוס/i.test(effectivePrompt)) &&
      lastSearchSources.some((s) => s.provider === "adsb-aviation" && s.ok && s.text.trim()) &&
      !!marineLiveCannedReply;
    if (pureAviationTurn && marineLiveCannedReply) {
      if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "ADS-B · canned reply")) return;
    }

    const pureShipsTurn =
      !qaForceLlm &&
      !wantsGameSearch &&
      !cameraActive &&
      !hasAttachments &&
      !continueCode &&
      !documentTurn &&
      isShipsQuery(effectivePrompt) &&
      !isMarineInfraQuery(effectivePrompt) &&
      lastSearchSources.some((s) => s.provider === "ais-ships" && s.ok && s.text.trim()) &&
      !!marineLiveCannedReply;
    if (pureShipsTurn && marineLiveCannedReply) {
      if (deliverLiveCannedReply(marineLiveCannedReply, webContext, "AIS · canned reply")) return;
    }

    const pureCapabilityLiveTurn =
      qaChatBridge.hasPending() &&
      structuredLiveTurn;
    if (pureCapabilityLiveTurn && marineLiveCannedReply) {
      if (finishCannedLive(marineLiveCannedReply, webContext)) return;
    }

    const wantsLongOutput =
      continueCode ||
      isCodeGenerationRequest(effectivePrompt) ||
      isCodeGenerationRequest(priorTurns.at(-1)?.content ?? "");
    let cameraImageBuffers: ArrayBuffer[] = [];
    let freshPersonBlock = "";
    let liveVisionSnapshot: VisionResult | null = null;

    if (cameraActive && cameraLoopRef.current && visionSensorQuery) {
      liveVisionSnapshot =
        cameraLoopRef.current.getLatestResult() ?? visionResultRef.current ?? null;
      try {
        setStatus("מכין דוח ראייה לפני תשובה…");
        const waitDeadline = Date.now() + 20_000;
        while (workerInferenceBusyRef.current && Date.now() < waitDeadline) {
          await new Promise((r) => setTimeout(r, 100));
        }
        if (personFocusRefresh) {
          const focus = await cameraLoopRef.current.refreshPersonFocus();
          if (focus) {
            freshPersonBlock = formatFreshPersonBlock(focus);
            pushActivity({
              direction: "system",
              kind: "person_focus",
              title: "Focus On Person · Vision Refresh",
              detail: freshPersonBlock,
            });
          }
        }
        const fresh = await cameraLoopRef.current.captureFreshSnapshot();
        cameraImageBuffers = [fresh];
        liveVisionSnapshot =
          cameraLoopRef.current.getLatestResult() ?? visionResultRef.current ?? null;
        pushActivity({
          direction: "system",
          kind: "vision_escalation",
          title: "Pre-chat snapshot + vision report",
          detail: [
            `"${trimmed}"`,
            `YOLO persons: ${liveVisionSnapshot?.objects.filter((o) => o.label === "person" && o.confidence >= 0.35).length ?? 0}`,
            `faces: ${liveVisionSnapshot?.faces.length ?? 0}`,
          ].join("\n"),
        });
      } catch {
        liveVisionSnapshot =
          cameraLoopRef.current.getLatestResult() ?? visionResultRef.current ?? null;
      }
    } else if (cameraActive) {
      pendingVisionContextRef.current = "";
      setStreamingVisionContext("");
    }

    const fingerCountBlock =
      fingerCountQuestion && liveVisionSnapshot
        ? appSettingsRef.current.vision.vision2Enabled
          ? buildFingerAnswerBlock(
              liveVisionSnapshot.fingerStates.reduce((sum, f) => sum + f.count, 0),
              liveVisionSnapshot.staticGestures.map((g) => g.name).join(", ") || undefined,
            )
          : buildFingerCountBlock(liveVisionSnapshot)
        : "";
    const attachmentBuffers =
      attachmentSnapshot.length > 0
        ? attachmentSnapshot.flatMap((p) => p.visionPages)
        : restoredImageBuffers;
    const ingestedDocBlock = buildIngestedDocumentPromptBlock(
      attachmentSnapshot.map((p) => ({
        kind: attachmentKindLabel(p.kind),
        label: p.label,
        extractedText: p.extractedText,
      })),
    );

    let documentOcrText = "";
    const skipOcr =
      hasSubstantialExtractedText(attachmentSnapshot) ||
      (attachmentBuffers.length === 0 && ingestedDocBlock.length > 0);
    if (documentTurn && attachmentBuffers.length && wantsExactTextExtraction(trimmed) && !skipOcr) {
      setStatus("קורא טקסט מהמסמך…");
      try {
        documentOcrText = await extractTextFromDocumentImages(attachmentBuffers, (msg) =>
          setStatus(msg),
        );
        if (documentOcrText.trim()) {
          pushActivity({
            direction: "system",
            kind: "document_ocr",
            title: "OCR · מסמך",
            detail: documentOcrText.slice(0, 1200),
          });
        }
      } catch {
        documentOcrText = "";
      }
    }

    const hasVisionInput = attachmentBuffers.length > 0 || cameraImageBuffers.length > 0;
    const factualCameraQuestion =
      personVisibilityQuestion ||
      personStateQuestion ||
      personDemographicsQuestion ||
      personMoodQuestion ||
      personActivityQuestion ||
      fingerCountQuestion ||
      visualDetailQuestion;
    const interpretiveCameraReply =
      cameraActive &&
      visionSensorQuery &&
      (hasVisionInput || liveCameraContext) &&
      !factualCameraQuestion &&
      !conversationFirst &&
      (sceneInterpretation || consciousnessQuestion || greetingWithCamera || greetingCameraStarting);
    const tokenBudgetBase =
      greetingWithCamera || greetingCameraStarting
        ? 100
        : interpretiveCameraReply
          ? Math.min(360, g.maxNewTokens)
          : greeting
            ? 40
            : documentTurn
              ? Math.min(1536, g.maxNewTokens)
              : hasVisionInput
                ? Math.min(1024, g.maxNewTokens)
                  : (shouldRunWebSearch || localTimeOnly) && webContext.trim()
                  ? Math.min(512, g.maxNewTokens)
                  : wantsLongOutput
                    ? Math.min(CODE_TOKEN_CAP, Math.max(g.maxNewTokens, CODE_TOKEN_FLOOR))
                    : g.maxNewTokens;
    let tokenBudget = tokenBudgetBase;

    let systemPrompt = cameraActive
      ? CAMERA_HAL_SYSTEM
      : greeting
        ? `${g.systemPrompt} If the user sends only a greeting, reply with one short warm sentence in their language only.`
        : g.systemPrompt;
    if (cameraActive) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_MODE_CHAT_APPEND}`;
    } else if (greetingWithCamera) {
      systemPrompt = `${g.systemPrompt}\n\n${CHARACTER_MODE_CHAT_APPEND}\n\n${GREETING_WITH_CAMERA_APPEND}`;
    } else if (greetingCameraStarting) {
      systemPrompt = `${g.systemPrompt}\n\n${CHARACTER_MODE_CHAT_APPEND}\n\n${GREETING_CAMERA_STARTING_APPEND}`;
    } else if (greeting) {
      systemPrompt = `${g.systemPrompt} If the user sends only a greeting, reply with one short warm sentence in their language only.`;
    }
    if (startupContext && !cameraActive) {
      systemPrompt = `${systemPrompt}\n\n${buildStartupPromptBlock(startupContext)}`;
    }
    if (cameraActive) {
      systemPrompt = `${systemPrompt}\n\n${CAMERA_ANTI_DEFLECT_APPEND}`;
      const relevantPast = findRelevantHistoryForPrompt(
        cameraStoreRef.current.messages,
        trimmed,
      );
      const memoryBlock = buildUserMemoryPromptBlock({
        profile: cameraStoreRef.current.profile,
        rollingSummary: cameraStoreRef.current.rollingSummary,
        relevantSnippets: relevantPast.length ? relevantPast : undefined,
      });
      systemPrompt = `${systemPrompt}\n\n${memoryBlock}`;
      if (cameraStoreRef.current.profile.name.trim()) {
        systemPrompt = `${systemPrompt}\n\nAddress the user as ${cameraStoreRef.current.profile.name.trim()} when natural (not every sentence).`;
      }
    }
    if (topicShifted && prevChatTopic) {
      systemPrompt = `${systemPrompt}\n\n${TOPIC_SHIFT_CHAT_APPEND}\n\n${topicShiftHint(prevChatTopic, chatTopic)}`;
      pushActivity({
        direction: "system",
        kind: "topic_shift",
        title: "שינוי נושא בצ'אט",
        detail: `${prevChatTopic} → ${chatTopic}: "${trimmed}"`,
      });
    }
    const searchHadLiveData = pendingWebSearchRef.current?.sources.some((s) => s.ok && s.text.trim()) ?? false;
    if (searchHadLiveData) {
      systemPrompt = `${systemPrompt}\n\n${buildWebSearchGroundingAppend({
        answerShape: pendingWebSearchRef.current?.answerShape,
        crossSource: pendingWebSearchRef.current?.crossSource,
      })}`;
    } else if (shouldRunWebSearch) {
      systemPrompt = `${systemPrompt}\n\n${WEB_SEARCH_NO_RESULTS_APPEND}`;
    }
    if (gameNoResults) {
      systemPrompt = `${systemPrompt}\n\n${GAME_SEARCH_NO_RESULTS_APPEND}`;
    } else if (gameGroundingBlock.trim()) {
      systemPrompt = `${systemPrompt}\n\n${GAME_SEARCH_GROUNDING_APPEND}\nGames found:\n${gameGroundingBlock}`;
    }
    if (globePlaceLabel) {
      systemPrompt = `${systemPrompt}\n\n${GLOBE_PRESENTATION_APPEND}\nPlace shown on map: ${globePlaceLabel}`;
    }
    if (cameraActive) {
      const vision2On = appSettingsRef.current.vision.vision2Enabled;
      const dialogueCtx = cameraLoopRef.current?.getDialogueContext() ?? null;
      const characterCtx = characterBrainRef.current;
      const characterBlock = `Character state: mood=${characterCtx.mood}, curiosity=${characterCtx.curiosity.toFixed(2)}, boredom=${characterCtx.boredom.toFixed(2)}`;
      if (visionSensorQuery) {
        const chatVisionBlock = buildChatVisionContextBlock(
          vision2On ? dialogueCtx : null,
          liveVisionSnapshot,
          worldMemoryRef.current,
          true,
          cameraImageBuffers.length > 0,
        );
        const internalVisionCtx = buildInternalVisionContextForUi({
          vision: liveVisionSnapshot,
          dialogue: vision2On ? dialogueCtx : null,
          world: worldMemoryRef.current,
          cameraActive: true,
          snapshotAttached: cameraImageBuffers.length > 0,
        });
        pendingVisionContextRef.current = internalVisionCtx;
        setStreamingVisionContext(internalVisionCtx);
        systemPrompt = `${systemPrompt}\n\n${CHAT_VISION_SYSTEM_HINT}\n\n${chatVisionBlock}\n\n${characterBlock}`;
        pushActivity({
          direction: "system",
          kind: "camera_context",
          title: "VISION QUERY → sensors",
          detail: internalVisionCtx.slice(0, 1600),
        });
      } else if (documentTurn) {
        pendingVisionContextRef.current = "";
        setStreamingVisionContext("");
        systemPrompt = `${systemPrompt}\n\n${characterBlock}`;
        pushActivity({
          direction: "system",
          kind: "camera_context",
          title: "DOCUMENT IMAGE (attached)",
          detail: `"${trimmed}"`,
        });
      } else {
        pendingVisionContextRef.current = "";
        setStreamingVisionContext("");
        systemPrompt = `${systemPrompt}\n\n${CAMERA_PURE_CHAT_APPEND}\n\n${characterBlock}`;
        pushActivity({
          direction: "system",
          kind: "camera_context",
          title: pureChatTurn ? "PURE CHAT (no sensors)" : "Dialogue (no vision pull)",
          detail: `"${trimmed}"`,
        });
      }
      if (freshPersonBlock) {
        systemPrompt = `${systemPrompt}\n\n${CURRENT_PERSON_STATE_APPEND}\n\n${freshPersonBlock}`;
      }
      if (personVisibilityQuestion || consciousnessQuestion || personDemographicsQuestion) {
        systemPrompt = `${systemPrompt}\n\n${PERSON_VISIBILITY_CHAT_APPEND}`;
      }
      if (personDemographicsQuestion) {
        systemPrompt = `${systemPrompt}\n\nUse faceData from INTERNAL VISION CONTEXT for gender and age. If faces=0 — person visible, demographics pending.`;
      }
      if (personStateQuestion) {
        systemPrompt = `${systemPrompt}\n\n${HOLDING_CHAT_APPEND}`;
      }
      if (personMoodQuestion) {
        systemPrompt = `${systemPrompt}\n\n${MOOD_CHAT_APPEND}`;
      }
      if (conversationFirst && !visionSensorQuery) {
        systemPrompt = `${systemPrompt}\n\n${CAMERA_CONVERSATION_APPEND}`;
      }
      if (greetingCameraStarting) {
        systemPrompt = `${systemPrompt}\n\n${GREETING_CAMERA_STARTING_APPEND}`;
      } else if (greetingWithCamera && visionSensorQuery) {
        systemPrompt = `${systemPrompt}\n\n${GREETING_WITH_CAMERA_APPEND}`;
      }
      if (interpretiveCameraReply && visionSensorQuery) {
        systemPrompt = `${systemPrompt}\n\n${CHARACTER_INTERPRETATION_APPEND}`;
      }
    }
    if (documentTurn) {
      systemPrompt = `${systemPrompt}\n\n${DOCUMENT_IMAGE_CHAT_APPEND}`;
      if (ingestedDocBlock.trim()) {
        systemPrompt = `${systemPrompt}\n\n[DOCUMENT TEXT — ground truth from file parser]\n${ingestedDocBlock.slice(0, 28_000)}`;
      }
      if (documentOcrText.trim()) {
        systemPrompt = `${systemPrompt}\n\n[OCR EXTRACT — verify against attached image]\n${documentOcrText.trim()}`;
      }
    } else if (hasAttachments && !cameraActive) {
      systemPrompt = `${systemPrompt} When the user sends an image, describe what you see accurately and answer their question in the same language as the user (Hebrew if they write in Hebrew).`;
    } else if (cameraActive && cameraImageBuffers.length && visionSensorQuery && !conversationFirst && !factualCameraQuestion) {
      systemPrompt = `${systemPrompt}\n\nSnapshot attached for your eyes only — do NOT describe it unless the user asked about vision.`;
    } else if (cameraImageBuffers.length && visualDetailQuestion) {
      systemPrompt = `${systemPrompt}\n\n${VISION_ESCALATION_CHAT_APPEND}`;
    } else if (cameraImageBuffers.length && personActivityQuestion) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_ACTIVITY_APPEND}`;
    } else if (cameraImageBuffers.length && (sceneInterpretation || greetingCameraStarting || greetingWithCamera)) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_INTERPRETATION_APPEND}`;
    } else if (personActivityQuestion && cameraActive) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_ACTIVITY_APPEND}`;
    } else if (personVisibilityQuestion && cameraActive && hasWorldData) {
      const peopleState = worldMemoryRef.current.people.length
        ? worldMemoryRef.current.people.join(", ")
        : "none — user not in frame";
      systemPrompt = `${systemPrompt}\n\n${PERSON_VISIBILITY_CHAT_APPEND}\nPeople in memory: ${peopleState}.`;
    } else if (fingerCountQuestion && cameraActive) {
      systemPrompt = `${systemPrompt}\n\n${FINGER_COUNT_CHAT_APPEND}\n\n${fingerCountBlock || "FINGER SENSOR: no fresh hand data — ask user to show hand clearly."}`;
      pushActivity({
        direction: "system",
        kind: "finger_count",
        title: "שאלת ספירת אצבעות",
        detail: fingerCountBlock || "(no hand data in latest frame)",
      });
    }
    if (continueCode) {
      systemPrompt = `${systemPrompt}\n\n${CONTINUE_CODE_SYSTEM_HINT}`;
      setStatus("ממשיך כתיבת קוד…");
    } else if (documentTurn) {
      setStatus("מנתח תמונה…");
    } else if (hasAttachments || (cameraActive && cameraImageBuffers.length && visionSensorQuery)) {
      setStatus(cameraActive ? "מנתח מצלמה…" : "מנתח תמונה…");
    } else if (cameraActive && pureChatTurn) {
      setStatus("חושב…");
    } else {
      setStatus(`Generating…${searchHint}${gameSearchHint}`);
    }

    systemPrompt = `${systemPrompt}\n\n${buildLanguageReplyDirective(effectivePrompt)}`;

    const currentImageBuffers = documentTurn
      ? attachmentBuffers
      : [...attachmentBuffers, ...cameraImageBuffers];

    const historyForWorkerRaw = trimHistoryForContext(priorTurns, undefined, continueCode);

    const chatProfileId = loadChatProfileOverride() ?? detectChatHardwareProfile();
    measuredSystemPromptCharsRef.current = systemPrompt.length;
    measuredWebContextCharsRef.current = webContext.length;
    setContextRefreshKey((k) => k + 1);
    const prepared = prepareChatContext({
      history: historyForWorkerRaw,
      webContext,
      systemPrompt,
      userPrompt: effectivePrompt,
      imageCount: currentImageBuffers.length,
      maxNewTokens: tokenBudget,
      profileId: chatProfileId,
      pinLastAssistant: continueCode,
      isSearchTurn: shouldRunWebSearch && !!webContext.trim(),
      isCodeTurn: wantsLongOutput || continueCode,
    });
    const historyForWorker = prepared.history;
    webContext = prepared.webContext;
    tokenBudget = prepared.maxNewTokens;

    const waitDeadline = Date.now() + 180_000;
    while (workerInferenceBusyRef.current && Date.now() < waitDeadline) {
      setStatus("ממתין לסיום ניתוח מצלמה…");
      await new Promise((r) => setTimeout(r, 250));
    }

    runGemmaGenerate(
      effectivePrompt,
      historyForWorker,
      systemPrompt,
      tokenBudget,
      greeting ? 0 : g.temperature,
      g.repetitionPenalty,
      g.topP,
      webContext,
      currentImageBuffers,
      "image/jpeg",
    );
    qaChatBridge.setWebContext(webContext);
    qaChatBridge.setReplySource("model");
  };

  const runCapabilitiesOnlyTurn = async (
    trimmed: string,
    priorChatMessages: ChatMessage[],
    effectivePrompt: string,
    chatTopic: ChatTopic,
  ) => {
    isGeneratingRef.current = true;
    setIsGenerating(true);
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setStatus("מחפש…");

    const priorTurns = buildHistoryForWorker(priorChatMessages).filter(
      (t) => t.role === "user" || t.role === "assistant",
    );

    const deliverCanned = (
      reply: string,
      webContext: string,
      replySource: string,
      activityTitle?: string,
    ) => {
      qaChatBridge.setWebContext(webContext);
      qaChatBridge.setReplySource(replySource);
      assistantBufferRef.current = reply;
      setAssistantBuffer(reply);
      setStatus("Ready");
      if (activityTitle) {
        pushActivity({
          direction: "system",
          kind: "web_search",
          title: activityTitle,
          detail: reply.slice(0, 1200),
        });
      }
      finalizeAssistantReply(false);
    };

    try {
      const preludeOutcome = await runTextChatTurnPrelude(
        {
          trimmed,
          effectivePrompt,
          priorTurns,
          chatTopic,
          startupContext,
          desktopLayout,
        },
        {
          setStatus,
          setStreamingSearchSources,
          setSearchResultsPayload,
          setSearchResultsOpen,
          setArtifactOpen,
          setGlobePanelOpen,
          setGlobeCommand,
          setGamesPanelOpen,
          setGamesPanelLayout,
          setGamesPanelGames,
          setGamesPanelTitle,
          setGamesPanelCategory,
          setGamesEmbedGame,
          setStreamingGameCategoryPicker,
          pushActivity,
          resolveSearchPlan: resolveSearchPlanForQuery,
          qaForceLlm: () => qaTurnForceLlmRef.current || qaChatBridge.isForceLlmPending(),
          qaHasPending: () => qaChatBridge.hasPending(),
          pendingWebSearchRef,
          pendingTimeWidgetRef,
          pendingGameCategoryPickerRef,
          pendingGameBrowseCategoryRef,
          deliverCanned,
        },
      );

      if (preludeOutcome.action === "canned") {
        qaChatBridge.notifyTurnComplete(assistantBufferRef.current);
        return;
      }

      const fallback = buildCapabilitiesOnlyFallbackMessage(
        capabilitiesFailureReason ?? undefined,
      );
      qaChatBridge.setWebContext(preludeOutcome.ctx.webContext);
      qaChatBridge.setReplySource("capabilities-only");
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: fallback },
      ]);
      setStatus("Ready");
      qaChatBridge.notifyTurnComplete(fallback);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(msg);
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: `⚠️ ${msg}` },
      ]);
      qaChatBridge.notifyTurnFailed(msg);
    } finally {
      isGeneratingRef.current = false;
      setIsGenerating(false);
      focusComposerInput();
    }
  };

  const runLocalTextTurn = async (trimmed: string, priorChatMessages: ChatMessage[]) => {
    const rack = applyLocalTextDownloadStates(modelRackRef.current, localTextDownloadingId);
    const model =
      getRackModelById(selectedRackModelRef.current, rack) ??
      getRackModelById(GEMMA_RACK_ID, rack)!;

    if (!model.hfModelId || model.status !== "ready") {
      setStatus("הורד את המודל מהבורר לפני שיחה");
      return;
    }

    isGeneratingRef.current = true;
    setIsGenerating(true);
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    streamTokenCountRef.current = 0;
    setStreamTokenCount(0);
    setStatus(`שיחה עם ${model.label}…`);

    const effectivePrompt = trimmed;
    const priorTurns = buildHistoryForWorker(priorChatMessages)
      .filter((t) => t.role === "user" || t.role === "assistant");
    const chatTopic = classifyChatTopic(trimmed);
    const uiLang = getChatUiLanguage();
    const bridgeHe = uiLang === "he";

    const deliverCanned = (
      reply: string,
      webContext: string,
      replySource: string,
      activityTitle?: string,
    ) => {
      qaChatBridge.setWebContext(webContext);
      qaChatBridge.setReplySource(replySource);
      assistantBufferRef.current = reply;
      setAssistantBuffer(reply);
      setStatus("Ready");
      if (activityTitle) {
        pushActivity({
          direction: "system",
          kind: "web_search",
          title: activityTitle,
          detail: reply.slice(0, 1200),
        });
      }
      finalizeAssistantReply(false);
    };

    try {
      const preludeOutcome = await runTextChatTurnPrelude(
        {
          trimmed,
          effectivePrompt,
          priorTurns,
          chatTopic,
          startupContext,
          desktopLayout,
        },
        {
          setStatus,
          setStreamingSearchSources,
          setSearchResultsPayload,
          setSearchResultsOpen,
          setArtifactOpen,
          setGlobePanelOpen,
          setGlobeCommand,
          setGamesPanelOpen,
          setGamesPanelLayout,
          setGamesPanelGames,
          setGamesPanelTitle,
          setGamesPanelCategory,
          setGamesEmbedGame,
          setStreamingGameCategoryPicker,
          pushActivity,
          resolveSearchPlan: resolveSearchPlanForQuery,
          qaForceLlm: () => qaTurnForceLlmRef.current || qaChatBridge.isForceLlmPending(),
          qaHasPending: () => qaChatBridge.hasPending(),
          pendingWebSearchRef,
          pendingTimeWidgetRef,
          pendingGameCategoryPickerRef,
          pendingGameBrowseCategoryRef,
          deliverCanned,
        },
      );

      if (preludeOutcome.action === "canned") {
        qaChatBridge.notifyTurnComplete(assistantBufferRef.current);
        return;
      }

      const { ctx: preludeCtx } = preludeOutcome;
      const statusSuffix = `${preludeCtx.searchHint}${preludeCtx.gameSearchHint}`;
      setStatus(`שיחה עם ${model.label}…${statusSuffix}`);

      const lt = appSettingsRef.current.localText;
      const history = priorTurns
        .slice(-lt.historyTurns)
        .map((t) => ({ role: t.role, content: t.content }));

      const baseSystem = buildLocalTextSystemPrompt({
        uiLang,
        prelude: preludeCtx,
        pendingWebSearch: pendingWebSearchRef.current,
        startupContext,
        webContext: preludeCtx.webContext,
        settings: lt,
      });

      if (bridgeHe) setStatus("מתרגם מעברית לאנגלית…");
      const prepared = await prepareLocalTextTurnForModel(trimmed, history, uiLang, baseSystem);
      setStatus(`שיחה עם ${model.label}…${statusSuffix}`);

      let modelReply = "";
      const maxTokens = localTextMaxNewTokens(preludeCtx, lt);
      const reply = await generateLocalTextChat({
        modelId: model.hfModelId,
        systemPrompt: prepared.systemPrompt,
        history: prepared.history,
        prompt: prepared.prompt,
        maxNewTokens: maxTokens,
        temperature: preludeCtx.greeting ? Math.min(lt.temperature, 0.55) : lt.temperature,
        topP: lt.topP,
        backend: lt.inferenceBackend,
        onToken: (text) => {
          modelReply += text;
          if (!bridgeHe) {
            assistantBufferRef.current += text;
            setAssistantBuffer((prev) => prev + text);
          }
        },
        onStatus: setStatus,
      });

      const rawEnglish = (reply.trim() || modelReply.trim() || assistantBufferRef.current.trim() || "").trim();
      if (bridgeHe) setStatus("מתרגם תשובה לעברית…");
      const content =
        (await localTextToUiLanguage(rawEnglish, uiLang)).trim() || rawEnglish || "…";

      qaChatBridge.setWebContext(preludeCtx.webContext);
      qaChatBridge.setReplySource("local-text");
      assistantBufferRef.current = content;
      setAssistantBuffer(content);
      finalizeAssistantReply(false);
      qaChatBridge.notifyTurnComplete(content);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(msg);
      const partial = assistantBufferRef.current.trim();
      if (partial) {
        const shown = bridgeHe ? await localTextToUiLanguage(partial, uiLang).catch(() => partial) : partial;
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "assistant", content: shown },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "assistant", content: `⚠️ ${msg}` },
        ]);
      }
      qaChatBridge.notifyTurnFailed(msg);
    } finally {
      isGeneratingRef.current = false;
      setIsGenerating(false);
      setAssistantBuffer("");
      assistantBufferRef.current = "";
      focusComposerInput();
    }
  };

  const runRackModelTurn = async (trimmed: string, effectivePrompt: string) => {
    const rack = modelRackRef.current;
    const model =
      getRackModelById(selectedRackModelRef.current, rack) ??
      getRackModelById(GEMMA_RACK_ID, rack)!;

    isGeneratingRef.current = true;
    setIsGenerating(true);
    setStatus(`מריץ ${model.label}…`);

    const result = await executeRackModel(model, trimmed || effectivePrompt, setStatus);

    isGeneratingRef.current = false;
    setIsGenerating(false);

    if (!result.ok) {
      setStatus(result.message);
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: `⚠️ ${result.message}` },
      ]);
      qaChatBridge.notifyTurnFailed(result.message);
      focusComposerInput();
      return;
    }

    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "assistant", content: result.content },
    ]);
    setStatus("Ready");
    qaChatBridge.setReplySource("rack");
    qaChatBridge.notifyTurnComplete(result.content);
    focusComposerInput();
  };

  const sendPrompt = async (e?: FormEvent, overrideText?: string) => {
    if (e) e.preventDefault();
    if (isGeneratingRef.current) {
      setStatus("עדיין עונה — המתן לסיום או לחץ עצור");
      qaChatBridge.notifyTurnFailed("busy");
      return;
    }
    const trimmed = (overrideText ?? prompt).trim();
    const attachmentSnapshot = pendingAttachments;
    const hasAttachments = attachmentSnapshot.length > 0;
    if (!trimmed && !hasAttachments) {
      qaChatBridge.notifyTurnFailed("empty prompt");
      return;
    }

    const activeRackModel =
      getRackModelById(
        selectedRackModelRef.current,
        applyLocalTextDownloadStates(modelRackRef.current, localTextDownloadingId),
      ) ?? getRackModelById(GEMMA_RACK_ID, modelRackRef.current)!;
    const usesLocalText =
      isLocalTextChatModel(activeRackModel) && !cameraModeRef.current;
    const usesExternalRack =
      rackModelRunsInChat(activeRackModel) && !cameraModeRef.current;

    if (usesLocalText && activeRackModel.status === "ready") {
      if (hasAttachments || cameraModeRef.current) {
        setStatus("SmolLM תומך רק בטקסט");
        qaChatBridge.notifyTurnFailed("attachments not supported");
        return;
      }
    } else if (
      !usesExternalRack &&
      !isGemmaLoadedRef.current &&
      chatModelAvailabilityRef.current !== "none"
    ) {
      setStatus("טען את Gemma מדף הפתיחה או בחר SmolLM");
      qaChatBridge.notifyTurnFailed("gemma not loaded");
      return;
    }

    cameraLoopRef.current?.holdForChat();
    const cameraActive = cameraModeRef.current;
    if (cameraActive) {
      characterBrainRef.current.recordUserInteraction();
    }
    generationCameraModeRef.current = cameraActive;
    const documentTurn = needsAttachedDocumentAnalysis(trimmed, hasAttachments);
    generationChatOnlyDocumentRef.current = documentTurn;
    setChatOnlyDocumentMode(documentTurn);

    const priorMessages = qaEmptyNextSendRef.current ? ((qaEmptyNextSendRef.current = false), []) : messages;

    const effectivePrompt =
      trimmed || defaultVisionPrompt(trimmed ? isRtlText(trimmed) : true);
    const chatTopic = classifyChatTopic(trimmed);

    const storedImages: StoredMessageImage[] = attachmentSnapshot
      .filter((p) => p.previewUrl)
      .map((p) => ({
        id: p.id,
        previewUrl: p.previewUrl!,
      }));
    for (const p of attachmentSnapshot) {
      p.visionPages.forEach((bytes, idx) => {
        const key = idx === 0 ? p.id : `${p.id}:${idx}`;
        imageBytesCacheRef.current.set(key, {
          bytes: bytes.slice(0),
          mime: "image/jpeg",
        });
      });
    }

    const attachmentLabels = attachmentSnapshot.map(
      (p) => `${attachmentKindLabel(p.kind)}: ${p.label}`,
    );
    const displayText =
      trimmed ||
      (hasAttachments
        ? attachmentLabels.length === 1
          ? `📄 ${attachmentLabels[0]}`
          : `📄 ${attachmentLabels.length} קבצים`
        : effectivePrompt);

    if (cameraActive) {
      setCameraStore((prev) => {
        const withUser: CameraSessionStore = {
          ...prev,
          updatedAt: Date.now(),
          memory: {
            ...prev.memory,
            topicLog: appendTopicToLog(prev.memory.topicLog, chatTopic),
          },
          messages: [
            ...prev.messages,
            {
              id: crypto.randomUUID(),
              role: "user",
              kind: "user",
              content: displayText,
              ts: Date.now(),
            },
          ],
        };
        const patched = patchCameraStoreAfterTurn(withUser, trimmed);
        saveCameraSessionStore(patched);
        return patched;
      });
    } else {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: displayText,
          images: storedImages.length ? storedImages : undefined,
        },
      ]);
    }
    setPrompt("");
    setPendingAttachments([]);
    setAttachError(null);
    focusComposerInput();

    const needsCapabilitiesPath =
      !usesExternalRack &&
      !cameraModeRef.current &&
      (chatModelAvailabilityRef.current === "none" ||
        (usesLocalText && activeRackModel.status !== "ready") ||
        (!usesLocalText && !isGemmaLoadedRef.current));

    if (needsCapabilitiesPath) {
      if (hasAttachments) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              "ללא מודל שיחה לא ניתן לנתח קבצים. בחר מודל תמונה בבורר ליצירה, או נסה חיפוש משחקים/חדשות.",
          },
        ]);
        qaChatBridge.notifyTurnFailed("no chat model attachments");
        return;
      }
      await runCapabilitiesOnlyTurn(trimmed, priorMessages, effectivePrompt, chatTopic);
      return;
    }

    if (usesLocalText) {
      await runLocalTextTurn(trimmed, priorMessages);
      return;
    }
    if (usesExternalRack) {
      await runRackModelTurn(trimmed, effectivePrompt);
      return;
    }
    await beginGeneration({
      trimmed,
      effectivePrompt,
      priorMessages,
      attachmentSnapshot,
      restoredImageBuffers: [],
    });
  };

  const sendPromptRef = useRef(sendPrompt);
  sendPromptRef.current = sendPrompt;

  const cancelMessageEdit = useCallback(() => {
    setEditingMessageId(null);
    setEditDraft("");
    focusComposerInput();
  }, [focusComposerInput]);

  const startMessageEdit = useCallback(
    (msg: ChatMessage) => {
      if (cameraMode || msg.role !== "user") return;
      if (isGenerating) {
        workerRef.current?.postMessage({ type: "abort" });
        isGeneratingRef.current = false;
        setIsGenerating(false);
        syncVisionBusy();
      }
      setEditingMessageId(msg.id);
      setEditDraft(msg.content);
    },
    [cameraMode, isGenerating, syncVisionBusy],
  );

  const submitMessageEdit = useCallback(async () => {
    if (!workerRef.current || !isLoaded || cameraMode || !editingMessageId) return;
    const trimmed = editDraft.trim();
    if (!trimmed) return;

    const idx = messages.findIndex((m) => m.id === editingMessageId);
    if (idx < 0 || messages[idx].role !== "user") return;

    if (isGenerating) {
      workerRef.current.postMessage({ type: "abort" });
      isGeneratingRef.current = false;
      setIsGenerating(false);
      syncVisionBusy();
      await new Promise((r) => setTimeout(r, 80));
    }

    const priorMessages = messages.slice(0, idx);
    const userMsg = messages[idx];
    const effectivePrompt = trimmed;
    const restoredImageBuffers = restoredImageBuffersFromMessage(userMsg);

    setMessages([...priorMessages, { ...userMsg, content: trimmed }]);
    setEditingMessageId(null);
    setEditDraft("");
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setStreamingSearchSources(null);
    focusComposerInput();

    await beginGeneration({
      trimmed,
      effectivePrompt,
      priorMessages,
      attachmentSnapshot: [],
      restoredImageBuffers,
    });
  }, [
    cameraMode,
    editDraft,
    editingMessageId,
    focusComposerInput,
    isGenerating,
    isLoaded,
    messages,
    restoredImageBuffersFromMessage,
    syncVisionBusy,
  ]);

  const sendActive = prompt.trim().length > 0 || pendingAttachments.length > 0;

  useEffect(() => {
    if (!QA_BRIDGE_ENABLED) return;
    exposeGroveeQaWindow();
    qaChatBridge.register({
      ready: () => isLoadedRef.current && !isGeneratingRef.current,
      newChat: () => {
        qaEmptyNextSendRef.current = true;
        qaTurnForceLlmRef.current = false;
        qaForceLlmRef.current = false;
        clearQueryCache();
        clearSharedRegionCache();
        isGeneratingRef.current = false;
        const id = newChatSessionId();
        setChatSessionsState((s) => ({
          activeId: id,
          sessions: [{ id, title: "שיחה חדשה", updatedAt: Date.now(), messages: [] }, ...s.sessions],
        }));
        setAssistantBuffer("");
        assistantBufferRef.current = "";
        setStreamingSearchSources(null);
        pendingWebSearchRef.current = null;
        pendingVisionContextRef.current = "";
        pendingTimeWidgetRef.current = null;
        pendingGameCategoryPickerRef.current = false;
        pendingGameBrowseCategoryRef.current = null;
        setPrompt("");
        setEditingMessageId(null);
        setEditDraft("");
        setArtifactOpen(false);
        setSearchResultsOpen(false);
        setSearchResultsPayload(null);
        setGlobePanelOpen(false);
        setGlobeCommand(null);
        setLiveMediaPanelOpen(false);
        setGamesPanelOpen(false);
        setIsGenerating(false);
      },
      submit: async (text, forceLlm) => {
        qaTurnForceLlmRef.current = forceLlm;
        qaForceLlmRef.current = forceLlm;
        await sendPromptRef.current(undefined, text);
      },
      getActivity: () => activityLogRef.current,
    });
    return () => qaChatBridge.unregister();
  }, []);

  return (
    <main className="app">
      {chatModelAvailability === "none" ? (
        <CapabilitiesWelcomeToast failureReason={capabilitiesFailureReason} />
      ) : null}

      {workerBootError && phase !== "ready" ? (
        <div className="worker-boot-banner" role="alert">
          <strong>שגיאה:</strong> {workerBootError}
          {isWebGpuInferenceError(workerBootError) ? (
            <button
              type="button"
              className="subtle-btn"
              style={{ marginInlineStart: 12 }}
              onClick={retryWasmLoad}
              disabled={isLoading}
            >
              טען ב-WASM
            </button>
          ) : null}
          <button
            type="button"
            className="subtle-btn"
            style={{ marginInlineStart: 12 }}
            onClick={() => setWorkerBootError(null)}
          >
            סגור
          </button>
        </div>
      ) : null}

      <SettingsModal
        key={settingsModalKey}
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={appSettings}
        onSave={persistSettings}
        onClearCache={() => void clearModelCache()}
        cacheClearing={cacheClearing}
        initialTab={settingsInitialTab}
      />

      <PluginsPanel
        open={pluginsOpen}
        onClose={() => setPluginsOpen(false)}
        tab={pluginsHubTab}
        onTabChange={setPluginsHubTab}
        healthSnapshot={pluginHealth}
        newsEngineStatus={newsEngineStatus}
        gemmaReady={isGemmaLoaded}
        gemmaLoading={isLoading}
        gemmaLoadPct={downloadProgressPercent(
          loadingBytes.loaded,
          loadingBytes.total > 0 ? loadingBytes.total : 0,
        )}
        gemmaLoadDetail={loadingByteLine}
        onRequestGemmaLoad={() => loadModel()}
      />

      <ModelActivityPanel
        open={activityLogOpen}
        onClose={() => setActivityLogOpen(false)}
        entries={activityLog}
        onClear={() => setActivityLog([])}
      />

      {QA_BRIDGE_ENABLED ? (
        <PresentationQaPanel
          open={presentationQaOpen}
          onClose={() => setPresentationQaOpen(false)}
          modelReady={isLoaded}
          isGenerating={isGenerating}
          activityLog={activityLog}
          assistantBuffer={assistantBuffer}
          streamingSearch={streamingSearchSources}
        />
      ) : null}

      {QA_VISION_MODE && !cameraMode ? (
        <video
          ref={cameraVideoRef}
          className="qa-vision-video"
          playsInline
          muted
          aria-hidden="true"
          style={{
            position: "fixed",
            width: 640,
            height: 480,
            top: 0,
            left: 0,
            opacity: 0.02,
            pointerEvents: "none",
            zIndex: 1,
          }}
        />
      ) : null}

      <VisionInspectorPanel
        open={visionInspectorOpen}
        onClose={() => setVisionInspectorOpen(false)}
        videoRef={cameraVideoRef}
        result={visionResult}
        config={visionPipelineConfig}
        onConfigChange={(partial) => {
          setVisionPipelineConfig((prev) => {
            const next = {
              ...prev,
              ...partial,
              toggles: { ...prev.toggles, ...partial.toggles },
              sampleIntervals: { ...prev.sampleIntervals, ...partial.sampleIntervals },
            };
            cameraLoopRef.current?.getPipeline().setConfig(next);
            savePipelineConfig(next);
            if (partial.performanceMode) {
              setAppSettings((s) => {
                const next = {
                  ...s,
                  vision: { ...s.vision, performanceMode: partial.performanceMode! },
                };
                saveSettings(next);
                return next;
              });
            }
            return next;
          });
        }}
        progress={visionPipelineProgress}
        cameraActive={cameraMode}
        showDetectionCards={appSettings.vision.showDetectionCards}
        worldMemory={worldMemorySnapshot}
      />

      <GroveeInfoModal open={infoModalOpen} onClose={() => setInfoModalOpen(false)} />

      {(phase === "start" || phase === "loading") && (
        <IntroScreen
          phase={phase}
          progress={progress}
          status={status}
          loadingPhase={loadingPhase}
          loadingByteLine={loadingByteLine}
          loadingFile={loadingFile}
          loadingTip={loadingTip}
          showWasmRetry={bootTarget === "gemma" && isWebGpuInferenceError(workerBootError ?? "")}
          cacheClearing={cacheClearing}
          isLoading={isLoading}
          isGenerating={isGenerating}
          onLoad={() => startIntroLoad()}
          onRetryWasm={retryWasmLoad}
          onOpenInfo={() => setInfoModalOpen(true)}
          onClearCache={() => void clearModelCache()}
          onContinueWithoutChat={() => enterCapabilitiesOnlyMode()}
          startupTarget={bootTarget}
          recommendedReasonHe={startupRecommendation?.reasonHe}
        />
      )}

      {phase === "ready" && (
        <div
          id="app-container"
          className={`app-container app-container--visible ${sidePanelBesideChat ? "app-container--artifact-open" : ""}${showLiveMediaFullscreen ? " app-container--livemedia-full" : ""}${showGamesFullscreen ? " app-container--games-full" : ""} ${sidebarOpen ? "app-container--sidebar-open" : ""}`}
        >
          <div
            className={`sb-overlay ${sidebarOpen ? "active" : ""}`}
            onClick={() => setSidebarOpen(false)}
            aria-hidden="true"
          />

          <aside className={`sidebar ${sidebarOpen ? "active" : ""}`}>
            {!sidebarOpen ? (
              <nav className="sidebar__rail" aria-label="תפריט צד">
                <button
                  type="button"
                  className="sb-rail-logo-btn"
                  aria-label="פתח תפריט GroVee"
                  title="פתח תפריט"
                  onClick={() => setSidebarOpen(true)}
                >
                  <GroveeLogoMark size="sm" />
                </button>
                <div className="sidebar__rail-actions">
                <button
                  type="button"
                  className="sb-rail-btn"
                  aria-label={cameraMode ? "נקה שיחת HAL" : "צ'אט חדש"}
                  title={cameraMode ? "נקה שיחת HAL" : "צ'אט חדש"}
                  onClick={handleNewChat}
                  disabled={isGenerating}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <path d="M12 20h9" />
                    <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
                  </svg>
                </button>
                <button
                  type="button"
                  className={`sb-rail-btn sb-rail-btn--search${showSearchResultsPanel ? " is-active" : ""}`}
                  aria-label="חיפוש GROVEE"
                  title="פתח חיפוש GROVEE"
                  aria-pressed={showSearchResultsPanel}
                  onClick={openSearchPanelFull}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <circle cx="11" cy="11" r="7" />
                    <path d="m20 20-3.5-3.5" />
                  </svg>
                </button>
                <button
                  type="button"
                  className={`sb-rail-btn sb-rail-btn--livemedia${showLiveMediaPanel ? " is-active" : ""}`}
                  aria-label="TV LIVE / רדיו"
                  title={showLiveMediaPanel ? "סגור TV LIVE / רדיו" : "פתח TV LIVE / רדיו"}
                  aria-pressed={showLiveMediaPanel}
                  onClick={toggleLiveMediaPanel}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <rect x="2" y="5" width="20" height="14" rx="2" />
                    <path d="M8 21h8" />
                    <path d="M12 19v2" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="sb-rail-btn sb-rail-btn--games"
                  aria-label="משחקים מומלצים"
                  title="פתח משחקים מומלצים"
                  onClick={() => void openGamesPanelFull()}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <line x1="6" y1="12" x2="10" y2="12" />
                    <line x1="8" y1="10" x2="8" y2="14" />
                    <line x1="15" y1="13" x2="15.01" y2="13" />
                    <line x1="18" y1="11" x2="18.01" y2="11" />
                    <path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="sb-rail-btn sb-rail-btn--globe"
                  aria-label="עולם חי"
                  title="פתח מוניטור עולם חי"
                  onClick={openGlobePanelFull}
                >
                  <span className="sb-globe-icon-wrap" aria-hidden="true">
                    <GlobeVisual size="xs" pulse tone="icon" />
                  </span>
                </button>
                </div>
                <div className="sidebar__rail-spacer" aria-hidden="true" />
                <SidebarGearMenu
                  variant="rail"
                  onSelect={handleGearMenuAction}
                  showPresentationQa={QA_BRIDGE_ENABLED}
                  visionDisabled={!cameraMode}
                  activityCount={activityLog.length}
                />
              </nav>
            ) : (
              <div className="sidebar__body">
                <div className="sb-header">
                  <button
                    type="button"
                    className="sb-header-logo-btn"
                    aria-label="כווץ תפריט"
                    title="כווץ תפריט"
                    onClick={() => setSidebarOpen(false)}
                  >
                    <GroveeLogoMark size="sm" />
                  </button>
                  <button
                    type="button"
                    className="sb-collapse-btn"
                    onClick={() => setSidebarOpen(false)}
                    aria-label="כווץ תפריט"
                    title="כווץ תפריט"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <path d="M9 3v18" />
                    </svg>
                  </button>
                </div>
                <div className="sb-sidebar-lang" dir="rtl">
                  <span className="sb-sidebar-lang-label">שפת ממשק</span>
                  <UiLanguageToggle className="sb-sidebar-lang-toggle" />
                </div>
                <div className="sidebar__scroll">
                <div className="sb-nav">
                  <button
                    type="button"
                    className="sb-nav-item sb-nav-item--primary"
                    onClick={handleNewChat}
                    disabled={isGenerating}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <path d="M12 20h9" />
                      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
                    </svg>
                    <span>{cameraMode ? "נקה שיחת HAL" : "צ'אט חדש"}</span>
                  </button>
                  <button
                    type="button"
                    className={`sb-nav-item sb-nav-item--search${showSearchResultsPanel ? " is-active" : ""}`}
                    onClick={openSearchPanelFull}
                    title="פתח חיפוש GROVEE"
                    aria-label="חיפוש GROVEE"
                    aria-pressed={showSearchResultsPanel}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <circle cx="11" cy="11" r="7" />
                      <path d="m20 20-3.5-3.5" />
                    </svg>
                    <span>חיפוש GROVEE</span>
                  </button>
                  <button
                    type="button"
                    className={`sb-nav-item sb-nav-item--livemedia${showLiveMediaPanel ? " is-active" : ""}`}
                    onClick={toggleLiveMediaPanel}
                    title={showLiveMediaPanel ? "סגור TV LIVE / רדיו" : "פתח TV LIVE / רדיו"}
                    aria-label="TV LIVE / רדיו"
                    aria-pressed={showLiveMediaPanel}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <rect x="2" y="5" width="20" height="14" rx="2" />
                      <path d="M8 21h8" />
                      <path d="M12 19v2" />
                    </svg>
                    <span>TV LIVE / רדיו</span>
                  </button>
                  <button
                    type="button"
                    className="sb-nav-item sb-nav-item--games"
                    onClick={() => void openGamesPanelFull()}
                    title="פתח משחקים מומלצים"
                    aria-label="משחקים מומלצים"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <line x1="6" y1="12" x2="10" y2="12" />
                      <line x1="8" y1="10" x2="8" y2="14" />
                      <line x1="15" y1="13" x2="15.01" y2="13" />
                      <line x1="18" y1="11" x2="18.01" y2="11" />
                      <path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z" />
                    </svg>
                    <span>משחקים מומלצים</span>
                  </button>
                  <button
                    type="button"
                    className="sb-nav-item sb-nav-item--globe"
                    onClick={openGlobePanelFull}
                    title="פתח מוניטור עולם חי"
                    aria-label="עולם חי"
                  >
                    <span className="sb-globe-icon-wrap sb-globe-icon-wrap--nav" aria-hidden="true">
                      <GlobeVisual size="xs" pulse tone="icon" />
                    </span>
                    <span>עולם חי</span>
                  </button>
                </div>
                {!cameraMode && visibleTextSessions.length > 0 ? (
                  <div className="sb-recents-label">אחרונות</div>
                ) : null}
                <div className="history chat-list">
                  {cameraMode ? (
                    <CameraUserProfilePanel
                      profile={cameraStore.profile}
                      rollingSummary={cameraStore.rollingSummary}
                      messageCount={cameraMessages.length}
                      searchHits={cameraSearchHits}
                      searchQuery={cameraHistorySearch}
                      onSearchChange={setCameraHistorySearch}
                      onSaveProfile={handleSaveCameraProfile}
                      disabled={isGenerating}
                    />
                  ) : (
                    visibleTextSessions.map((s) => (
                    <div
                      key={s.id}
                      className={`hist-row ${s.id === chatSessionsState.activeId ? "active" : ""}`}
                    >
                      <button
                        type="button"
                        className="hist-item chat-item"
                        title={s.title}
                        onClick={() => {
                          if (s.id === chatSessionsState.activeId || isGenerating) return;
                          setChatSessionsState((st) => ({ ...st, activeId: s.id }));
                          setAssistantBuffer("");
                          assistantBufferRef.current = "";
                          setEditingMessageId(null);
                          setEditDraft("");
                          setArtifactOpen(false);
                        }}
                        disabled={isGenerating}
                      >
                        {s.title}
                      </button>
                      <button
                        type="button"
                        className="hist-delete-btn"
                        aria-label={`מחק שיחה: ${s.title}`}
                        title="מחק שיחה"
                        disabled={isGenerating}
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteChatSession(s.id);
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))
                  )}
                </div>
                </div>
                <div className="user-foot">
                  <ChatMessageAvatar role="user" className="avatar" />
                  <span>אורח</span>
                  <SidebarGearMenu
                    variant="footer"
                    onSelect={handleGearMenuAction}
                    showPresentationQa={QA_BRIDGE_ENABLED}
                    visionDisabled={!cameraMode}
                    activityCount={activityLog.length}
                  />
                </div>
              </div>
            )}
          </aside>

          {anySidePanelOpen ? (
            <aside
              className={`artifact-panel side-panel open ${showCameraSidePanel ? "side-panel--camera" : ""}${showGamesSidePanel ? " side-panel--games" : ""}${showGamesFullscreen ? " side-panel--games-full" : ""}${showGlobePanel ? " side-panel--globe" : ""}${showSearchResultsPanel ? " side-panel--search" : ""}${showLiveMediaPanel ? " side-panel--livemedia" : ""}${showLiveMediaFullscreen ? " side-panel--livemedia-full" : ""}`}
              aria-label={
                showLiveMediaPanel
                  ? "TV LIVE / רדיו"
                  : showSearchResultsPanel
                  ? "תוצאות חיפוש"
                  : showArtifactPanel
                  ? "חלונית קוד"
                  : showGamesPanel
                    ? "משחקים און־ליין"
                    : showGlobePanel
                      ? "מוניטור עולמי"
                      : "מצלמה חיה"
              }
            >
              {showLiveMediaPanel ? (
                <LiveMediaPanel uiLang="he" onClose={closeLiveMediaPanel} layout={desktopLayout ? "full" : "side"} />
              ) : showSearchResultsPanel && searchResultsPayload ? (
                <SearchResultsPanel
                  payload={searchResultsPayload}
                  onClose={closeSearchResultsPanel}
                  onSearch={handleSearchPanelQuery}
                  searching={searchPanelLoading}
                  onSummaryReady={handleNewsArticlePolish}
                  onHfAddedToRack={handleRackUpdated}
                />
              ) : showArtifactPanel ? (
                <ArtifactPanel
                  artifact={activeArtifact!}
                  streaming={isGenerating && !!assistantBuffer}
                  streamTokenCount={streamTokenCount}
                  onClose={() => setArtifactOpen(false)}
                />
              ) : showGamesPanel ? (
                <GamesPanel
                  games={gamesPanelGames}
                  loading={gamesPanelLoading}
                  embedGame={gamesEmbedGame}
                  title={gamesPanelTitle}
                  initialCategory={gamesPanelCategory}
                  startView={gamesPanelStartView}
                  layout={gamesPanelLayout}
                  onExpandFull={expandGamesPanelFull}
                  onShrinkSide={shrinkGamesPanelSide}
                  onClose={closeGamesPanel}
                  onPlay={handlePlayGame}
                  onBackFromEmbed={() => setGamesEmbedGame(null)}
                  onGamesUpdate={handleGamesPanelUpdate}
                  onLoadingChange={setGamesPanelLoading}
                />
              ) : showGlobePanel ? (
                <GlobePanel
                  onClose={closeGlobePanel}
                  command={globeCommand}
                  onCommandSent={() => setGlobeCommand(null)}
                  modelReady={isLoaded}
                />
              ) : (
                <CameraPreview
                  ref={cameraVideoRef}
                  variant="panel"
                  active={cameraMode}
                  observing={cameraObserving}
                  mood={characterMood}
                  hal={halMoodState}
                  interpretation={halInterpretation}
                  consciousness={halConsciousness}
                  entity={halEntity}
                  cameraStatus={cameraStatus}
                  error={cameraError}
                  visionResult={visionResult}
                  pipelineConfig={visionPipelineConfig}
                  pipelineProgress={visionPipelineProgress}
                  visionPaused={isGenerating}
                  onVideoReady={(video) => void startVisionPipeline(video)}
                  onPipelineConfigChange={onCameraPipelineConfigChange}
                />
              )}
            </aside>
          ) : null}

          <section
            className={`chat-area ${showLanding ? "chat-area--landing" : ""}${showLiveMediaFullscreen ? " chat-area--hidden-livemedia" : ""}${showGamesFullscreen ? " chat-area--hidden-games" : ""}`}
            aria-hidden={showLiveMediaFullscreen || showGamesFullscreen}
          >
            <header className="chat-header">
              <button
                type="button"
                className="chat-header-menu-btn"
                aria-label="פתח תפריט GroVee"
                title="פתח תפריט"
                onClick={() => setSidebarOpen(true)}
              >
                <GroveeLogoMark size="sm" />
              </button>
              <LocalContextBar
                context={startupContext}
                uiLang={uiLang}
                variant="header"
                className="chat-header-context-mobile"
              />
              <div className="chat-header-primary">
              {!cameraMode ? (
                <ChatModelPicker
                  rack={pickerModelRack}
                  selectedId={selectedRackModelId}
                  onSelect={handleRackModelSelect}
                  onDownloadLocalText={(entry) => void handleDownloadLocalText(entry)}
                  downloadState={{
                    downloadingId: localTextDownloadingId,
                    progressPct: localTextDownloadPct,
                    progressLabel: localTextDownloadLabel,
                  }}
                  disabled={isGenerating}
                />
              ) : (
                <div className="camera-mode-header" dir="rtl">
                  <span className="camera-mode-title">🎥 שיחת מצלמה</span>
                  <span className="camera-mode-sub">
                    זיכרון נפרד · {cameraMessages.length} הודעות
                  </span>
                </div>
              )}
              </div>
              <div className="chat-header-actions">
                <UiLanguageToggle className="chat-header-lang-desktop" />
                <LocalContextBar
                  context={startupContext}
                  uiLang={uiLang}
                  className="chat-header-context-desktop"
                />
                {activeArtifact && !artifactOpen ? (
                  <button
                    type="button"
                    className="artifact-reopen-btn"
                    onClick={() => setArtifactOpen(true)}
                  >
                    פתח {activeArtifact.kind === "html" ? "HTML" : "קוד"}
                  </button>
                ) : null}
                {cameraMode ? (
                  <button
                    type="button"
                    className="activity-log-btn"
                    title="מחק זיכרון שיחת מצלמה (הודעות + היכרות)"
                    disabled={isGenerating}
                    onClick={() => {
                      if (isGenerating) return;
                      const fresh = clearCameraSessionStore();
                      setCameraStore(fresh);
                      characterBrainRef.current.reset();
                    }}
                  >
                    🗑 זיכרון מצלמה
                  </button>
                ) : null}
              </div>
            </header>

            <div className="chat-body">
              {cameraMode ? <CameraTopicBar topics={cameraStore.memory.topicLog} /> : null}
              {!showLanding ? (
                <div className="msg-list-scroll" ref={messagesListRef}>
                  <div className="msg-list messages">
                  {cameraMode
                    ? cameraMessages.map((msg) => (
                        <article
                          key={msg.id}
                          className={`msg${msg.kind === "proactive" ? " msg--proactive" : ""}`}
                          dir={isRtlText(msg.content) ? "rtl" : "ltr"}
                        >
                          <ChatMessageAvatar
                            role={msg.role === "user" ? "user" : "assistant"}
                            variant="hal"
                          />
                          <div className="msg-txt">
                            {msg.kind === "proactive" ? (
                              <span className="msg-proactive-tag">יוזמה</span>
                            ) : null}
                            <MessageBody
                              content={msg.content}
                              onOpenArtifact={openArtifact}
                              savedThought={msg.thought}
                              savedVisionContext={msg.visionContext}
                            />
                          </div>
                        </article>
                      ))
                    : messages.map((msg) => (
                        <article
                          key={msg.id}
                          className={`msg${msg.role === "user" ? " msg--user" : ""}`}
                          dir={isRtlText(msg.content) ? "rtl" : "ltr"}
                        >
                          <ChatMessageAvatar role={msg.role === "user" ? "user" : "assistant"} />
                          <div className="msg-txt">
                            {msg.role === "user" ? (
                              <>
                                {msg.images?.length ? <UserAttachedImages images={msg.images} /> : null}
                                <ChatUserMessage
                                  isEditing={editingMessageId === msg.id}
                                  editDraft={editDraft}
                                  canEdit
                                  isRtl={isRtlText(msg.content)}
                                  onStartEdit={() => startMessageEdit(msg)}
                                  onCancelEdit={cancelMessageEdit}
                                  onDraftChange={setEditDraft}
                                  onSaveEdit={() => void submitMessageEdit()}
                                >
                                  <MessageBody
                                    content={msg.content}
                                    onOpenArtifact={openArtifact}
                                    savedThought={msg.thought}
                                    savedVisionContext={msg.visionContext}
                                  />
                                </ChatUserMessage>
                              </>
                            ) : (
                              <>
                                {msg.artifact ? (
                                  <ArtifactChip
                                    kind={msg.artifact.kind}
                                    label={msg.artifact.kind === "html" ? "HTML" : msg.artifact.title}
                                    onOpen={() => openArtifact(msg.artifact!)}
                                  />
                                ) : null}
                                {msg.searchSources?.length ? (
                                  <SearchProgressPanel
                                    active={false}
                                    sources={msg.searchSources}
                                    summary={msg.searchSummary}
                                  />
                                ) : null}
                                {msg.timeWidget ? <TimeClockWidget data={msg.timeWidget} /> : null}
                                {msg.showGameCategories ? (
                                  <GameCategoryPicker
                                    activeCategory={msg.gameBrowseCategory}
                                    onPick={(cat) => void handleGameCategoryPick(cat)}
                                    onOpenFavorites={openGamesFavorites}
                                  />
                                ) : null}
                                <MessageBody
                                  content={msg.content}
                                  onOpenArtifact={openArtifact}
                                  savedThought={msg.thought}
                                  savedVisionContext={msg.visionContext}
                                />
                              </>
                            )}
                          </div>
                        </article>
                      ))}
                  {assistantBuffer && (
                    <article className="msg">
                      <ChatMessageAvatar role="assistant" variant={cameraMode ? "hal" : "default"} />
                      <div className="msg-txt">
                        {streamingSearchSources ? (
                          <SearchProgressPanel
                            active={!!streamingSearchSources.active}
                            query={streamingSearchSources.query}
                            sources={streamingSearchSources.sources}
                            summary={streamingSearchSources.summary}
                            brief={streamingSearchSources.brief}
                          />
                        ) : null}
                        {streamingGameCategoryPicker ? (
                          <GameCategoryPicker
                            activeCategory={gamesPanelCategory}
                            onPick={(cat) => void handleGameCategoryPick(cat)}
                            onOpenFavorites={openGamesFavorites}
                          />
                        ) : null}
                        <MessageBody
                          content={assistantBuffer}
                          onOpenArtifact={openArtifact}
                          showThinking={thinkingMode}
                          savedVisionContext={streamingVisionContext}
                          chatOnlyDocument={chatOnlyDocumentMode}
                        />
                      </div>
                    </article>
                  )}
                  </div>
                </div>
              ) : (
                <div className="chat-landing-spacer" aria-hidden="true" />
              )}

              <div className="chat-composer-stack">
                {showLanding ? <ChatLandingHeadline text={landingContent.headline} /> : null}

                <div className="composer-modes">
              {isLoaded && contextUsage && !showLanding ? (
                <ContextRing usage={contextUsage} />
              ) : null}
              {showCameraInline ? (
                <CameraPreview
                  ref={cameraVideoRef}
                  active={cameraMode}
                  observing={cameraObserving}
                  mood={characterMood}
                  hal={halMoodState}
                  interpretation={halInterpretation}
                  consciousness={halConsciousness}
                  entity={halEntity}
                  cameraStatus={cameraStatus}
                  error={cameraError}
                  visionResult={visionResult}
                  pipelineConfig={visionPipelineConfig}
                  pipelineProgress={visionPipelineProgress}
                  visionPaused={isGenerating}
                  onVideoReady={(video) => void startVisionPipeline(video)}
                  onPipelineConfigChange={onCameraPipelineConfigChange}
                />
              ) : null}
              {cameraStatus && !showLanding ? (
                <span className="composer-status-hint composer-status-hint--camera">{cameraStatus}</span>
              ) : null}
              {!showLanding && status && status !== "Not loaded" && status !== "Ready" ? (
                <span className="composer-status-hint">{status}</span>
              ) : null}
            </div>

            <form
              className={`input-zone ${isDragOver ? "input-zone--drag" : ""} ${showLanding ? "input-zone--landing" : ""}`}
              onSubmit={sendPrompt}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={(e) => {
                if (e.currentTarget.contains(e.relatedTarget as Node)) return;
                setIsDragOver(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragOver(false);
                if (e.dataTransfer.files?.length) void addFilesAsAttachments(e.dataTransfer.files);
              }}
            >
              {pendingAttachments.length > 0 ? (
                <div className="composer-attachments">
                  {pendingAttachments.map((p) => (
                    <div key={p.id} className="composer-attachment">
                      {p.previewUrl ? (
                        <img src={p.previewUrl} alt="" className="composer-attachment-thumb" />
                      ) : (
                        <div className="composer-attachment-doc" title={p.label}>
                          <span className="composer-attachment-doc-icon" aria-hidden="true">
                            📄
                          </span>
                          <span className="composer-attachment-doc-label">{attachmentKindLabel(p.kind)}</span>
                        </div>
                      )}
                      <button
                        type="button"
                        className="composer-attachment-remove"
                        onClick={() => removePendingAttachment(p.id)}
                        aria-label="הסר קובץ"
                        disabled={isGenerating || attachProcessing}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              ) : null}
              {attachError ? (
                <p className="composer-attach-error" role="alert">
                  {attachError}
                </p>
              ) : null}
              <input
                ref={fileInputRef}
                type="file"
                accept={DOCUMENT_ACCEPT}
                multiple
                hidden
                onChange={(e) => {
                  if (e.target.files?.length) void addFilesAsAttachments(e.target.files);
                  e.target.value = "";
                }}
              />
              <div className="in-box">
                <ComposerPlusMenu
                  attachDisabled={
                    !isLoaded ||
                    isGenerating ||
                    attachProcessing ||
                    pendingAttachments.length >= MAX_ATTACHMENTS
                  }
                  onAttachClick={() => fileInputRef.current?.click()}
                  thinkingMode={thinkingMode}
                  onThinkingToggle={() => {
                    if (isGenerating) return;
                    setThinkingMode((v) => !v);
                  }}
                  thinkingDisabled={isGenerating}
                  cameraMode={cameraMode}
                  onCameraToggle={() => void toggleCameraMode()}
                  cameraDisabled={isGenerating || (!isGemmaLoaded && !QA_VISION_MODE)}
                />
                <textarea
                  ref={textareaRef}
                  id="user-in"
                  dir="auto"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onPaste={(e) => {
                    const items = e.clipboardData?.items;
                    if (!items) return;
                    const imageFiles: File[] = [];
                    for (const item of items) {
                      if (item.type.startsWith("image/")) {
                        const f = item.getAsFile();
                        if (f) imageFiles.push(f);
                      }
                    }
                    if (imageFiles.length) {
                      e.preventDefault();
                      void addFilesAsAttachments(imageFiles);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing) return;
                    e.preventDefault();
                    e.currentTarget.form?.requestSubmit();
                  }}
                  placeholder={
                    cameraMode
                      ? "דבר עם GROVEE…"
                      : pendingAttachments.length
                        ? "שאל על התמונה…"
                        : "הקלד הודעה…"
                  }
                  rows={1}
                  disabled={!isLoaded}
                />
                {isGenerating ? (
                  <button
                    type="button"
                    className="in-act in-stop"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={stopGeneration}
                    aria-label="עצור"
                    title="עצור יצירה"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  </button>
                ) : (
                  <>
                    <ComposerVoiceMic
                      disabled={!isLoaded || isGenerating}
                      onTranscript={appendVoiceTranscript}
                    />
                    <button
                      type="submit"
                      className={`in-act in-send ${sendActive ? "in-send--active" : ""}`}
                      onMouseDown={(e) => e.preventDefault()}
                      disabled={!isLoaded || !sendActive || isGenerating}
                      aria-label="שלח"
                      title="שלח"
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path
                          d="M12 19V5"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="m7 10 5-5 5 5"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                  </>
                )}
              </div>

                {showLanding ? (
                  <ChatLandingSuggestions
                    suggestions={landingContent.suggestions}
                    rotationKey={landingContent.rotationKey}
                    onSuggestionClick={applyLandingSuggestion}
                  />
                ) : null}
            </form>
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
