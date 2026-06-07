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
  needsCameraVisionEscalation,
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
  prepareImageAttachment,
  type PendingImage,
  type StoredMessageImage,
} from "./imageAttachments";
import { IntroCanvas } from "./IntroCanvas";
import { ArtifactPanel, type Artifact } from "./ArtifactPanel";
import {
  buildPersistedAssistantPayload,
  extractPrimaryArtifact,
  extractRichParts,
} from "./artifacts";
import { formatBytes, requestPersistentStorage } from "./storageReport";
import {
  SCENE_ANALYSIS_SYSTEM_PROMPT,
  buildSceneAnalysisUserPrompt,
} from "./cameraPrompts";
import {
  CAMERA_ANTI_DEFLECT_APPEND,
  CAMERA_CHAT_WORLD_HINT,
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
} from "./characterPrompts";
import { buildProactiveSensorBlock, buildRichSensorBlock, poseFromWorld } from "./sensorBlock";
import { buildFingerCountBlock } from "./visionBridge";
import { pauseTfVision, resumeTfVision } from "./visionCoordination";
import { formatFreshPersonBlock } from "./personFocus";
import {
  attachStreamToVideo,
  requestCameraStream,
  type CameraStreamHandle,
} from "./cameraMode";
import { checkBrowserVisionSupport } from "./browserVision";
import { detectVisionBudget } from "./visionBudget";
import { CharacterBrain, moodLabelHe, type CharacterMood } from "./characterBrain";
import { WorldMemory } from "./worldMemory";
import {
  GroveeVisionRunner,
  type CharacterDecision,
  type SceneAnalysisResult,
} from "./GroveeVisionRunner";
import { CameraPreview } from "./CameraPreview";
import { ModelActivityPanel } from "./ModelActivityPanel";
import { VisionInspectorPanel } from "./VisionInspectorPanel";
import {
  ensureVisionLabConfig,
  loadPipelineConfig,
  savePipelineConfig,
} from "./vision-lab/core/configStorage";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";
import { appendModelActivity, type ModelActivityEntry } from "./modelActivityLog";

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
  /** Thumbnails for user-attached images (session memory; not persisted to localStorage). */
  images?: StoredMessageImage[];
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
    };

const GEMMA_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const SETTINGS_STORAGE_KEY = "grovee_model_settings_v1";
const CHATS_STORAGE_KEY = "grovee_chats_v1";

/** Friendly product tips while Gemma downloads — explain what GROVEE is. */
const LOADING_DOWNLOAD_TIPS = [
  "GROVEE הוא צ'אט AI חינמי — רץ על המחשב שלך, בלי מנוי ובלי שליחת שיחות לענן.",
  "שואלים בעברית או באנגלית, מקבלים תשובות ישירות מהדפדפן — המודל Gemma 4 E2B נטען אצלך.",
  "הפרטיות נשארת אצלך: מה שאתה כותב לא עובר לשרת AI חיצוני.",
  "אחרי שההורדה הראשונה תסתיים, אפשר לשוחח גם בלי חיבור אינטרנט.",
  "אפשר לבקש קוד, הסברים, סיפורים ודפי HTML — GROVEE מייצר הכול מקומית.",
  "זו לא אפליקציית ענן: המשקולות (~3.9GB, פעם ראשונה) נשמרות במטמון הדפדפן.",
  "GROVEE בנוי על Transformers.js — AI בדפדפן, כולל ראייה (תמונות) מקומית.",
  "צרף תמונה בכפתור 📎 או הדבק (Ctrl+V) — המודל יתאר ויפענח אותה אצלך במחשב.",
  "כפתור Think מפעיל <|think|> native של Gemma 4; Search מוסיף הקשר מוויקיפדיה ו-GitHub.",
  "ההורדה ארוכה רק בפעם הראשונה — בפעם הבאה GROVEE יעלה הרבה יותר מהר.",
  "עוד רגע תוכל לפתוח שיחה חדשה ולדבר עם העוזר המקומי שלך — חינם לגמרי.",
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
};

const defaultGemmaSettings: TunableModelSettings = {
  temperature: 0.2,
  maxNewTokens: 2048,
  repetitionPenalty: 1.12,
  topP: 0.9,
  systemPrompt:
    'You are a helpful assistant. Always respond in clear, well-formed sentences in the same language as the user (Hebrew stays RTL-friendly: full sentences, correct punctuation at end of sentence). Do not repeat role labels. When the user asks for HTML/CSS/JS (including a single-file page), output exactly one fenced block: ```html ... ``` containing a complete, valid document: <!DOCTYPE html>, <html lang="he" dir="rtl">, <head> with <meta charset="UTF-8">, embedded <style> and <script> as needed, and <body>. No duplicate stray tags; no broken CSS.',
};

const defaultAppSettings = (): AppSettings => ({
  hfRemoteHost: "",
  inferenceBackend: "auto",
  gemma: { ...defaultGemmaSettings },
});

const loadSettings = (): AppSettings => {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return defaultAppSettings();
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      ...defaultAppSettings(),
      hfRemoteHost: typeof parsed.hfRemoteHost === "string" ? parsed.hfRemoteHost : "",
      inferenceBackend:
        parsed.inferenceBackend === "webgpu" || parsed.inferenceBackend === "wasm" || parsed.inferenceBackend === "auto"
          ? parsed.inferenceBackend
          : "auto",
      gemma: { ...defaultGemmaSettings, ...parsed.gemma },
    };
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

const defaultChatSessionsState = (): ChatSessionsState => {
  const id = newChatSessionId();
  return {
    activeId: id,
    sessions: [{ id, title: "שיחה חדשה", updatedAt: Date.now(), messages: [] }],
  };
};

const loadChatSessionsState = (): ChatSessionsState => {
  try {
    const raw = localStorage.getItem(CHATS_STORAGE_KEY);
    if (!raw) return defaultChatSessionsState();
    const parsed = JSON.parse(raw) as { activeId?: string; sessions?: ChatSession[] };
    if (!parsed.sessions || !Array.isArray(parsed.sessions) || parsed.sessions.length === 0) {
      return defaultChatSessionsState();
    }
    const sessions = parsed.sessions.map((s) => ({
      id: typeof s.id === "string" ? s.id : newChatSessionId(),
      title: typeof s.title === "string" ? s.title : "שיחה",
      updatedAt: typeof s.updatedAt === "number" ? s.updatedAt : Date.now(),
      messages: Array.isArray(s.messages) ? s.messages : [],
    }));
    const activeId =
      parsed.activeId && sessions.some((x) => x.id === parsed.activeId) ? parsed.activeId : sessions[0].id;
    return { activeId, sessions };
  } catch {
    return defaultChatSessionsState();
  }
};

const saveChatSessionsState = (state: ChatSessionsState) => {
  const serializable = { activeId: state.activeId, sessions: state.sessions };
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

const WEB_LOOKUP_USER_AGENT = "GROVEEMODEL/1.0 (browser chat; no backend)";

const expandHebrewTechSearchTerms = (query: string): string => {
  const parts: string[] = [];
  if (/מצלמ[ות]? אבטחה|מצלמת אבטחה|אבטחה ומעקב/i.test(query)) {
    parts.push("security", "camera", "surveillance");
  }
  if (/ממשק|דשבורד|ניהול/i.test(query)) {
    parts.push("dashboard", "interface", "ui");
  }
  if (/ניטור|הקלטה|הקלטות/i.test(query)) {
    parts.push("monitoring", "recording");
  }
  if (/קוד\s*פתוח/i.test(query)) {
    parts.push("open", "source");
  }
  return [...new Set(parts)].join(" ").trim();
};

const buildGitHubSearchQuery = (query: string): string => {
  const raw = query.trim();
  if (!raw) return "";
  const latinTokens = raw.match(/[a-zA-Z][a-zA-Z0-9_.-]{1,}/g);
  const latin = latinTokens ? latinTokens.join(" ") : "";
  if (latin.length >= 3) return latin.slice(0, 256);
  const wantsGithub = /github|גיטהב/i.test(raw);
  const hebrewHints = expandHebrewTechSearchTerms(raw);
  if (wantsGithub && hebrewHints) return hebrewHints.slice(0, 256);
  if (wantsGithub && latin.length > 0) return latin.slice(0, 256);
  if (hebrewHints.length >= 8) return hebrewHints.slice(0, 256);
  return "";
};

const fetchWikipediaSnippets = async (query: string, lang: "en" | "he"): Promise<string> => {
  const encoded = encodeURIComponent(query);
  const endpoint = `https://${lang}.wikipedia.org/w/api.php?action=opensearch&search=${encoded}&limit=4&namespace=0&format=json&origin=*`;
  try {
    const response = await fetch(endpoint);
    if (!response.ok) return "";
    const data = (await response.json()) as [string, string[], string[], string[]];
    const titles = data[1] ?? [];
    const snippets = data[2] ?? [];
    const urls = data[3] ?? [];
    if (!titles.length) return "";
    return titles
      .map((title, i) => `- ${title}: ${snippets[i] ?? ""} (${urls[i] ?? ""})`)
      .join("\n");
  } catch {
    return "";
  }
};

const fetchGitHubRepoHits = async (searchQuery: string): Promise<string> => {
  const q = searchQuery.trim();
  if (!q) return "";
  const url = `https://api.github.com/search/repositories?q=${encodeURIComponent(q)}&sort=stars&order=desc&per_page=6`;
  try {
    const response = await fetch(url, {
      headers: {
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": WEB_LOOKUP_USER_AGENT,
      },
    });
    if (!response.ok) return "";
    const data = (await response.json()) as {
      items?: Array<{ full_name: string; description: string | null; html_url: string; stargazers_count: number }>;
    };
    const items = data.items ?? [];
    if (!items.length) return "";
    return items
      .map(
        (item) =>
          `- ${item.full_name}${item.description ? `: ${item.description}` : ""} (${item.html_url}) ★${item.stargazers_count}`,
      )
      .join("\n");
  } catch {
    return "";
  }
};

const fetchWebContext = async (query: string): Promise<string> => {
  const q = query.trim();
  if (!q) return "";
  const hasHebrew = /[\u0590-\u05FF]/.test(q);
  const ghq = buildGitHubSearchQuery(q);

  const [wikiEn, wikiHe, github] = await Promise.all([
    fetchWikipediaSnippets(q, "en"),
    hasHebrew ? fetchWikipediaSnippets(q, "he") : Promise.resolve(""),
    ghq ? fetchGitHubRepoHits(ghq) : Promise.resolve(""),
  ]);

  const blocks: string[] = [];
  if (wikiEn) blocks.push(`Wikipedia (en):\n${wikiEn}`);
  if (wikiHe) blocks.push(`Wikipedia (he):\n${wikiHe}`);
  if (github) blocks.push(`GitHub repositories:\n${github}`);

  return blocks.join("\n\n");
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
  savedArtifact,
}: {
  content: string;
  onOpenArtifact?: (artifact: Artifact) => void;
  showThinking?: boolean;
  savedThought?: string;
  savedArtifact?: Artifact;
}) {
  const streamParts = useMemo(() => splitAssistantStream(content, showThinking), [content, showThinking]);
  const thoughtText = showThinking ? streamParts.thought : (savedThought ?? "");
  const displayContent = showThinking ? streamParts.answer : content;
  const parts = useMemo(() => extractRichParts(displayContent), [displayContent]);
  const dir = isRtlText(displayContent || thoughtText) ? "rtl" : "ltr";

  return (
    <div className="msg-body" dir={dir}>
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
          <p key={i} className="msg-text">
            {part.value}
          </p>
        );
      })}
    </div>
  );
}

function SettingsModal({
  open,
  onClose,
  settings,
  onSave,
  onClearCache,
  cacheClearing,
}: {
  open: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSave: (s: AppSettings) => void;
  onClearCache: () => void;
  cacheClearing: boolean;
}) {
  const [draft, setDraft] = useState<AppSettings>(() => settings);

  if (!open) return null;

  const setBackend = (inferenceBackend: InferenceBackendPreference) => {
    setDraft((d) => ({ ...d, inferenceBackend }));
  };

  const backendOptions: { id: InferenceBackendPreference; label: string; hint: string }[] = [
    { id: "auto", label: "Auto", hint: "WebGPU אם אפשר, אחרת WASM" },
    { id: "wasm", label: "WASM", hint: "מעבד — יציב" },
    { id: "webgpu", label: "WebGPU", hint: "GPU — מהיר יותר" },
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
            <span className="settings-head-badge">G</span>
            <div>
              <h2 id="settings-title">הגדרות Gemma</h2>
              <p className="settings-head-sub">GEMMA 4 E2B · מקומי בדפדפן</p>
            </div>
          </div>
          <button type="button" className="icon-close settings-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </div>

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

        <div className="settings-footer">
          <button type="button" className="settings-btn-ghost" onClick={() => setDraft(defaultAppSettings())}>
            איפוס ברירת מחדל
          </button>
          <button
            type="button"
            className="settings-btn-save"
            onClick={() => {
              onSave(draft);
              onClose();
            }}
          >
            שמור
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const workerRef = useRef<Worker | null>(null);
  const assistantBufferRef = useRef("");
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
  const workerInferenceBusyRef = useRef(false);
  const sceneAnalysisMetaRef = useRef(new Map<string, { reason?: string }>());

  const [appSettings, setAppSettings] = useState<AppSettings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsModalKey, setSettingsModalKey] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState("Not loaded");
  const [progress, setProgress] = useState(0);
  const [loadingPhase, setLoadingPhase] = useState<"download" | "init">("download");
  const [loadingBytes, setLoadingBytes] = useState({ loaded: 0, total: 0, speedBps: 0 });
  const [loadingTipIndex, setLoadingTipIndex] = useState(0);
  const [workerBootError, setWorkerBootError] = useState<string | null>(null);
  const [workerReloadKey, setWorkerReloadKey] = useState(0);
  const [cacheClearing, setCacheClearing] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<PendingImage[]>([]);
  const [attachError, setAttachError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [chatSessionsState, setChatSessionsState] = useState<ChatSessionsState>(() => loadChatSessionsState());
  const [assistantBuffer, setAssistantBuffer] = useState("");
  const [thinkingMode, setThinkingMode] = useState(false);
  const [webSearchMode, setWebSearchMode] = useState(false);
  const [cameraMode, setCameraMode] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraObserving, setCameraObserving] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("");
  const [characterMood, setCharacterMood] = useState<CharacterMood>("observing");
  const [activityLogOpen, setActivityLogOpen] = useState(false);
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
  }));
  const [visionPipelineConfig, setVisionPipelineConfig] = useState<PipelineConfig>(() =>
    loadPipelineConfig(),
  );
  const [activityLog, setActivityLog] = useState<ModelActivityEntry[]>([]);
  const [infoModalOpen, setInfoModalOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [artifactOpen, setArtifactOpen] = useState(false);
  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null);

  const appSettingsRef = useRef(appSettings);
  const thinkingRef = useRef(thinkingMode);
  const webSearchRef = useRef(webSearchMode);
  const cameraModeRef = useRef(cameraMode);
  const isLoadingRef = useRef(isLoading);
  const isGeneratingRef = useRef(isGenerating);
  const messagesListRef = useRef<HTMLDivElement | null>(null);
  const continueModeRef = useRef(false);
  const loadingFileRef = useRef("");

  useEffect(() => {
    appSettingsRef.current = appSettings;
  }, [appSettings]);
  useEffect(() => {
    thinkingRef.current = thinkingMode;
  }, [thinkingMode]);
  useEffect(() => {
    webSearchRef.current = webSearchMode;
  }, [webSearchMode]);
  useEffect(() => {
    cameraModeRef.current = cameraMode;
  }, [cameraMode]);
  useEffect(() => {
    isLoadingRef.current = isLoading;
  }, [isLoading]);
  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  useEffect(() => {
    saveChatSessionsState(chatSessionsState);
  }, [chatSessionsState]);

  const activeSession = useMemo(
    () =>
      chatSessionsState.sessions.find((s) => s.id === chatSessionsState.activeId) ?? chatSessionsState.sessions[0],
    [chatSessionsState],
  );
  const messages = activeSession.messages;

  const sortedSessions = useMemo(
    () => [...chatSessionsState.sessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [chatSessionsState.sessions],
  );

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
  const modelLabel = "Gemma 4 E2B";

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
  }, [messages, assistantBuffer, phase]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el || phase !== "ready") return;
    el.style.height = "auto";
    el.style.height = `${Math.min(Math.max(el.scrollHeight, 36), 120)}px`;
  }, [prompt, pendingAttachments.length, phase]);

  const buildHistoryForWorker = useCallback((priorMessages: ChatMessage[]): ChatTurn[] => {
    return priorMessages.map((m) => {
      if (m.role !== "user" || !m.images?.length) {
        return { role: m.role, content: m.content };
      }
      const images = m.images
        .map((img) => imageBytesCacheRef.current.get(img.id))
        .filter((x): x is { bytes: ArrayBuffer; mime: string } => !!x && x.bytes.byteLength > 0)
        .map((x) => ({ bytes: x.bytes, mime: x.mime }));
      return { role: m.role, content: m.content, images: images.length ? images : undefined };
    });
  }, []);

  const addFilesAsAttachments = useCallback(async (files: FileList | File[]) => {
    setAttachError(null);
    const list = Array.from(files).filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|gif)$/i.test(f.name));
    if (!list.length) {
      setAttachError("לא נבחרו קבצי תמונה");
      return;
    }
    const room = MAX_ATTACHMENTS - pendingAttachments.length;
    if (room <= 0) {
      setAttachError(`מקסימום ${MAX_ATTACHMENTS} תמונות`);
      return;
    }
    const slice = list.slice(0, room);
    try {
      const prepared = await Promise.all(slice.map((f) => prepareImageAttachment(f)));
      setPendingAttachments((prev) => [...prev, ...prepared]);
    } catch (err) {
      setAttachError(err instanceof Error ? err.message : String(err));
    }
  }, [pendingAttachments.length]);

  const removePendingAttachment = useCallback((id: string) => {
    setPendingAttachments((prev) => {
      const target = prev.find((p) => p.id === id);
      if (target) URL.revokeObjectURL(target.previewUrl);
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
        const sensorBlock = buildProactiveSensorBlock(world, poseFromWorld(world), {
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

  const stopCameraMode = useCallback(() => {
    cameraBootingRef.current = false;
    cameraLoopRef.current?.dispose();
    cameraLoopRef.current = null;
    cameraStreamRef.current?.stop();
    cameraStreamRef.current = null;
    setCameraStream(null);
    worldMemoryRef.current.reset();
    characterBrainRef.current.reset();
    setCharacterMood("observing");
    setCameraMode(false);
    setCameraObserving(false);
    setCameraError(null);
    setCameraStatus("");
    sceneAnalysisResolversRef.current.forEach((resolve) => resolve(null));
    sceneAnalysisResolversRef.current.clear();
    characterUtteranceResolversRef.current.forEach((resolve) => resolve(null));
    characterUtteranceResolversRef.current.clear();
  }, []);

  const toggleCameraMode = useCallback(async () => {
    if (cameraMode) {
      stopCameraMode();
      return;
    }
    if (!isLoaded || isGenerating) return;
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
  }, [cameraMode, isGenerating, isLoaded, stopCameraMode]);

  const startVisionPipeline = useCallback(
    async (video: HTMLVideoElement) => {
      if (!cameraStream || !isLoaded || cameraBootingRef.current || cameraLoopRef.current) return;
      cameraBootingRef.current = true;
      try {
        await attachStreamToVideo(video, cameraStream);

        cameraLoopRef.current?.dispose();
        worldMemoryRef.current.reset();
        characterBrainRef.current.reset();
        chatTopicRef.current = null;

        const visionBudget = detectVisionBudget();
        cameraVisionBudgetRef.current = visionBudget;
        const labConfig = ensureVisionLabConfig(
          visionPipelineConfig,
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
              if (!b.useLlmProactiveUtterance) return false;
              if (cameraLoopRef.current?.isDeepVisionDegraded()) return false;
              return true;
            },
            resolveUtterance: async (decision) => {
              const llm = await requestCharacterUtterance(decision);
              if (llm?.trim()) return { ...decision, message: llm.trim() };
              return decision;
            },
            isWorkerBusy: () => isGeneratingRef.current || workerInferenceBusyRef.current,
            onCameraStatus: setCameraStatus,
            onMoodChange: setCharacterMood,
            onPipelineProgress: setVisionPipelineProgress,
            onVisionResult: (result) => setVisionResult({ ...result }),
            onCharacterSpeak: (decision: CharacterDecision) => {
              pushActivity({
                direction: "in",
                kind: "character_speak",
                title: `Character · ${decision.mood}`,
                detail: `${decision.reason}\n\n${decision.message}`,
                meta: { topic: decision.topic, mood: decision.mood },
              });
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: decision.message,
                  modelLabel: `HAL · ${moodLabelHe(decision.mood)}`,
                },
              ]);
            },
            onObservingChange: setCameraObserving,
            onLightDetection: (payload) => {
              const evLines = payload.worldUpdate.newEvents.map(
                (e) => `[${e.type}] ${e.text}${e.subject ? ` (${e.subject})` : ""}`,
              );
              pushActivity({
                direction: "system",
                kind: "light_detect",
                title: "Vision Lab · YOLO",
                detail: [
                  `objects: ${payload.objects.join(", ") || "(none)"}`,
                  `personCount(raw): ${payload.rawPeople}`,
                  `people(debounced): ${payload.debouncedPeople.join(", ") || "(none)"}`,
                  payload.personJustConfirmed ? "person debounce: CONFIRMED in frame" : "",
                  payload.personJustLeft ? "person debounce: LEFT frame" : "",
                  payload.worldUpdate.isBaselineCapture ? "baseline capture (no events)" : "",
                  payload.worldUpdate.suppressedAsChurn ? "suppressed: camera churn" : "",
                  evLines.length ? `events:\n${evLines.join("\n")}` : "",
                ]
                  .filter(Boolean)
                  .join("\n"),
                meta: {
                  personCount: payload.rawPeople,
                  personConfirmed: payload.debouncedPeople.length > 0,
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
          },
          visionBudget,
          labConfig,
        );
        runner.bindVideo(video);
        cameraLoopRef.current = runner;
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
    if (!cameraMode || !cameraStream || !isLoaded) return;
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
    if (phase !== "ready" && cameraMode) stopCameraMode();
  }, [phase, cameraMode, stopCameraMode]);

  useEffect(() => {
    if (cameraMode && !isGenerating) {
      syncVisionBusy();
    }
  }, [cameraMode, isGenerating, syncVisionBusy]);

  const finalizeAssistantReply = useCallback(
    (stopped: boolean) => {
      isGeneratingRef.current = false;
      setIsGenerating(false);
      syncVisionBusy();
      const raw = assistantBufferRef.current;
      const { content, artifact, thought } = raw.trim()
        ? buildPersistedAssistantPayload(raw, thinkingRef.current)
        : { content: "", artifact: null, thought: undefined };

      if (!content.trim() && !artifact) {
        setAssistantBuffer("");
        assistantBufferRef.current = "";
        setStatus(stopped ? "התשובה נעצרה" : "Ready");
        continueModeRef.current = false;
        cameraLoopRef.current?.releaseAfterChat();
        return;
      }

      if (continueModeRef.current) {
        continueModeRef.current = false;
        setMessages((prev) => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i--) {
            if (next[i].role === "assistant") {
              next[i] = {
                ...next[i],
                content: artifact ? next[i].content || content : content,
                artifact: artifact ?? next[i].artifact,
                thought: thought ?? next[i].thought,
              };
              return next;
            }
          }
          return [
            ...next,
            {
              id: crypto.randomUUID(),
              role: "assistant" as const,
              content,
              artifact: artifact ?? undefined,
              thought,
              modelLabel: "Gemma 4",
            },
          ];
        });
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content,
            artifact: artifact ?? undefined,
            thought,
            modelLabel: "Gemma 4",
          },
        ]);
      }

      if (artifact) {
        setActiveArtifact(artifact);
      }

      setAssistantBuffer("");
      assistantBufferRef.current = "";
      setStatus(stopped ? "התשובה נעצרה" : "Ready");
      cameraLoopRef.current?.releaseAfterChat();
    },
    [setMessages, syncVisionBusy],
  );

  const stopGeneration = useCallback(() => {
    if (!isGenerating) return;
    workerRef.current?.postMessage({ type: "abort" });
    setStatus("עוצר…");
  }, [isGenerating]);

  useEffect(() => {
    if (phase !== "ready") return;
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
      if (msg.type === "status") {
        setStatus(msg.text);
      } else if (msg.type === "progress") {
        setStatus(msg.text);
        if (isLoadingRef.current) {
          if (msg.phase === "init") {
            setProgress(msg.progress);
            loadingFileRef.current = "";
            setLoadingBytes({ loaded: 0, total: 0, speedBps: 0 });
          } else {
            const nextFile = msg.file ?? "";
            const nextLoaded = typeof msg.loaded === "number" ? msg.loaded : 0;
            const nextTotal = typeof msg.total === "number" ? msg.total : 0;
            const nextSpeed = typeof msg.speedBps === "number" ? msg.speedBps : 0;

            if (nextFile) loadingFileRef.current = nextFile;
            setProgress((prev) => Math.max(prev, msg.progress));
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
        setIsLoaded(true);
        setIsLoading(false);
        setProgress(100);
        setStatus(`Gemma ready on ${formatInferenceDevice(msg.device)}`);
      } else if (msg.type === "token") {
        setAssistantBuffer((prev) => {
          const next = prev + msg.text;
          assistantBufferRef.current = next;
          return next;
        });
      } else if (msg.type === "done") {
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
          setStatus(`שגיאה: ${msg.error}`);
          cameraLoopRef.current?.releaseAfterChat();
        } else {
          setIsLoading(false);
          setProgress(0);
          setStatus(`Error: ${msg.error}`);
          setWorkerBootError(msg.error);
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
      } else if (msg.type === "character_utterance") {
        workerInferenceBusyRef.current = false;
        const resolveUtterance = characterUtteranceResolversRef.current.get(msg.requestId);
        if (resolveUtterance) {
          characterUtteranceResolversRef.current.delete(msg.requestId);
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

  const loadModel = () => {
    if (!workerRef.current) return;
    setWorkerBootError(null);
    setIsLoading(true);
    setIsLoaded(false);
    setStatus("Loading Gemma 4 E2B…");
    setProgress(0);
    setLoadingPhase("download");
    setLoadingBytes({ loaded: 0, total: 0, speedBps: 0 });
    setLoadingTipIndex(0);
    loadingFileRef.current = "";
    workerRef.current.postMessage({
      type: "configure_hub",
      remoteHost: appSettingsRef.current.hfRemoteHost ?? "",
    });
    workerRef.current.postMessage({
      type: "configure_inference",
      backend: appSettingsRef.current.inferenceBackend,
    });
    workerRef.current.postMessage({
      type: "load",
      modelId: GEMMA_MODEL_ID,
      dtype: "q4",
    });
  };

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
      setIsLoading(false);
      setAssistantBuffer("");
      assistantBufferRef.current = "";
      setWorkerReloadKey((k) => k + 1);

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
    setAppSettings((prev) => {
      if (s.inferenceBackend !== prev.inferenceBackend || s.hfRemoteHost !== prev.hfRemoteHost) {
        queueMicrotask(() => {
          setIsLoaded(false);
          setIsLoading(false);
          setStatus("הגדרות השתנו — לחץ «התחל» כדי לטעון מחדש");
        });
      }
      return s;
    });
    saveSettings(s);
  };

  const sendPrompt = async (e: FormEvent) => {
    e.preventDefault();
    if (!workerRef.current || !isLoaded || isGenerating) return;
    const trimmed = prompt.trim();
    const hasImages = pendingAttachments.length > 0;
    if (!trimmed && !hasImages) return;

    const liveVisionSnapshot = cameraModeRef.current
      ? cameraLoopRef.current?.getLatestResult() ?? null
      : null;

    cameraLoopRef.current?.holdForChat();
    characterBrainRef.current.recordUserInteraction();

    const prevChatTopic = chatTopicRef.current;
    const chatTopic = classifyChatTopic(trimmed);
    const topicShifted = isTopicShift(prevChatTopic, chatTopic);
    chatTopicRef.current = chatTopic;

    const effectivePrompt =
      trimmed || defaultVisionPrompt(trimmed ? isRtlText(trimmed) : true);

    const priorTurns = buildHistoryForWorker(messages);

    const continueCode = shouldContinueCode(effectivePrompt, priorTurns);
    continueModeRef.current = continueCode;

    const storedImages: StoredMessageImage[] = pendingAttachments.map((p) => ({
      id: p.id,
      previewUrl: p.previewUrl,
    }));
    for (const p of pendingAttachments) {
      imageBytesCacheRef.current.set(p.id, { bytes: p.modelBytes.slice(0), mime: p.mime });
    }

    const displayText = trimmed || (hasImages ? "🖼️ תמונה" : effectivePrompt);

    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: displayText,
        images: storedImages.length ? storedImages : undefined,
      },
    ]);
    setPrompt("");
    setPendingAttachments([]);
    setAttachError(null);

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

    isGeneratingRef.current = true;
    setIsGenerating(true);
    syncVisionBusy();

    const g = appSettings.gemma;
    const greeting = isSimpleGreeting(effectivePrompt) && !hasImages;
    const cameraActive = cameraModeRef.current;
    const fingerCountQuestion = isFingerCountQuestion(trimmed);
    const fingerCountBlock =
      fingerCountQuestion && liveVisionSnapshot
        ? buildFingerCountBlock(liveVisionSnapshot)
        : "";
    const hasWorldData = worldMemoryRef.current.hasData();
    const greetingWithCamera = greeting && cameraActive && hasWorldData;
    const greetingCameraStarting = greeting && cameraActive && !hasWorldData;
    const visionEscalation = needsCameraVisionEscalation(trimmed);
    const visualDetailQuestion = isVisualDetailQuestion(trimmed);
    const personVisibilityQuestion = isPersonVisibilityQuestion(trimmed);
    const personActivityQuestion = isPersonActivityQuestion(trimmed);
    const personStateQuestion = isCurrentPersonStateQuestion(trimmed);
    const personFocusRefresh = needsPersonFocusRefresh(trimmed);
    const sceneInterpretation = isSceneInterpretationQuestion(trimmed);
    const cameraChatVision =
      visionEscalation || greetingCameraStarting || (personFocusRefresh && cameraActive);
    let webContext = "";
    let searchHint = "";
    if (webSearchMode && !hasImages) {
      setStatus("Searching…");
      try {
        webContext = await fetchWebContext(effectivePrompt);
        if (!webContext.trim()) searchHint = " · אין תוצאות חיפוש";
      } catch {
        webContext = "";
        searchHint = " · חיפוש נכשל";
      }
    }

    const wantsLongOutput =
      continueCode ||
      isCodeGenerationRequest(effectivePrompt) ||
      isCodeGenerationRequest(priorTurns.at(-1)?.content ?? "");
    let cameraImageBuffers: ArrayBuffer[] = [];
    let freshPersonBlock = "";
    if (cameraActive && cameraLoopRef.current && (cameraChatVision || personFocusRefresh)) {
      try {
        const waitDeadline = Date.now() + 30_000;
        while (
          (workerInferenceBusyRef.current || isGeneratingRef.current) &&
          Date.now() < waitDeadline
        ) {
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
              detail: [
                freshPersonBlock,
                focus.poseState !== "unknown"
                  ? `pose_change check: ${worldMemoryRef.current.lastChanges
                      .filter((e) => (e.subject ?? "").startsWith("pose_change"))
                      .slice(0, 2)
                      .map((e) => e.subject ?? e.text)
                      .join(" | ") || "none recent"}`
                  : "",
              ]
                .filter(Boolean)
                .join("\n\n"),
            });
          }
        }
        const fresh = await cameraLoopRef.current.captureFreshSnapshot();
        cameraImageBuffers = [fresh];
        const title = personStateQuestion
          ? "שאלת מצב נוכחי — snapshot + pose refresh"
          : visualDetailQuestion
            ? "Smart Vision Escalation"
            : personVisibilityQuestion
              ? "זיהוי נוכחות — snapshot"
              : greetingCameraStarting
                ? "ברכה + snapshot סצנה"
                : "צילום מצלמה לצ'אט";
        pushActivity({
          direction: "system",
          kind: "vision_escalation",
          title,
          detail: personStateQuestion
            ? `שאלת תנוחה/מצב — refresh + snapshot: "${trimmed}"`
            : personVisibilityQuestion
              ? `שאלת נראות — snapshot + people=${worldMemoryRef.current.people.join(",") || "none"}: "${trimmed}"`
              : greetingCameraStarting
                ? `ברכה עם מצלמה — snapshot: "${trimmed}"`
                : visualDetailQuestion
                  ? `שאלת פרט חזותי — snapshot חדש: "${trimmed}"`
                  : `שאלת הקשר מצלמה — snapshot חדש: "${trimmed}"`,
        });
      } catch {
        /* camera not ready */
      }
    }
    const hasVisionInput = hasImages || cameraImageBuffers.length > 0;
    const interpretiveCameraReply =
      cameraActive &&
      hasVisionInput &&
      !visualDetailQuestion &&
      (sceneInterpretation || personActivityQuestion || personVisibilityQuestion || personStateQuestion || greetingWithCamera || greetingCameraStarting);
    const tokenBudget =
      greetingWithCamera || greetingCameraStarting
        ? 100
        : interpretiveCameraReply
          ? Math.min(360, g.maxNewTokens)
          : greeting
            ? 40
            : hasVisionInput
              ? Math.min(1024, g.maxNewTokens)
              : wantsLongOutput
                ? Math.min(CODE_TOKEN_CAP, Math.max(g.maxNewTokens, CODE_TOKEN_FLOOR))
                : g.maxNewTokens;

    let systemPrompt = greetingWithCamera
      ? `${g.systemPrompt}\n\n${CHARACTER_MODE_CHAT_APPEND}\n\n${GREETING_WITH_CAMERA_APPEND}`
      : greetingCameraStarting
        ? `${g.systemPrompt}\n\n${CHARACTER_MODE_CHAT_APPEND}\n\n${GREETING_CAMERA_STARTING_APPEND}`
        : greeting
          ? `${g.systemPrompt} If the user sends only a greeting, reply with one short friendly sentence only.`
          : g.systemPrompt;
    if (cameraActive && !greetingWithCamera && !greetingCameraStarting) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_MODE_CHAT_APPEND}`;
    }
    if (cameraActive) {
      systemPrompt = `${systemPrompt}\n\n${CAMERA_ANTI_DEFLECT_APPEND}`;
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
    if (cameraActive && worldMemoryRef.current.hasData()) {
      const richSensors = buildRichSensorBlock(worldMemoryRef.current, visionResult);
      const cameraBlock = freshPersonBlock
        ? `${worldMemoryRef.current.toCharacterAtmosphereBlock()}\n\nSensor layers:\n${richSensors}`
        : `${worldMemoryRef.current.toCharacterChatBlock()}\n\nSensor layers:\n${richSensors}`;
      const characterCtx = characterBrainRef.current;
      const characterBlock = `Character state: mood=${characterCtx.mood}, curiosity=${characterCtx.curiosity.toFixed(2)}, boredom=${characterCtx.boredom.toFixed(2)}`;
      systemPrompt = `${systemPrompt}\n\n${CAMERA_CHAT_WORLD_HINT}\n\n${cameraBlock}\n\n${characterBlock}`;
      if (freshPersonBlock) {
        systemPrompt = `${systemPrompt}\n\n${CURRENT_PERSON_STATE_APPEND}\n\n${freshPersonBlock}`;
      }
      pushActivity({
        direction: "system",
        kind: "camera_context",
        title: greetingWithCamera ? "ברכה + הקשר סצנה" : "הקשר Character (interpretation)",
        detail: `${cameraBlock}\n\n${characterBlock}`,
      });
    }
    if (hasImages) {
      systemPrompt = `${systemPrompt} When the user sends an image, describe what you see accurately and answer their question in the same language as the user (Hebrew if they write in Hebrew).`;
    } else if (cameraImageBuffers.length && visualDetailQuestion) {
      systemPrompt = `${systemPrompt}\n\n${VISION_ESCALATION_CHAT_APPEND}`;
    } else if (cameraImageBuffers.length && personStateQuestion) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_INTERPRETATION_APPEND}`;
    } else if (cameraImageBuffers.length && personActivityQuestion) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_ACTIVITY_APPEND}\n\n${CHARACTER_INTERPRETATION_APPEND}`;
    } else if (cameraImageBuffers.length && personVisibilityQuestion) {
      systemPrompt = `${systemPrompt}\n\n${PERSON_VISIBILITY_CHAT_APPEND}\n\n${CHARACTER_INTERPRETATION_APPEND}`;
    } else if (cameraImageBuffers.length && (sceneInterpretation || greetingCameraStarting || greetingWithCamera)) {
      systemPrompt = `${systemPrompt}\n\n${CHARACTER_INTERPRETATION_APPEND}`;
    } else if (cameraImageBuffers.length && cameraActive) {
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
    } else if (hasImages || (cameraActive && cameraImageBuffers.length)) {
      setStatus(cameraActive ? "מנתח מצלמה…" : "מנתח תמונה…");
    } else {
      setStatus(`Generating…${searchHint}`);
    }

    const historyForWorker = trimHistoryForContext(priorTurns, 32_000, continueCode);

    const currentImageBuffers = [
      ...storedImages
        .map((img) => imageBytesCacheRef.current.get(img.id)?.bytes)
        .filter((b): b is ArrayBuffer => !!b),
      ...cameraImageBuffers,
    ];

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
  };

  const sendActive = prompt.trim().length > 0 || pendingAttachments.length > 0;

  return (
    <main className="app">
      {workerBootError ? (
        <div className="worker-boot-banner" role="alert">
          <strong>שגיאה:</strong> {workerBootError}
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
      />

      <ModelActivityPanel
        open={activityLogOpen}
        onClose={() => setActivityLogOpen(false)}
        entries={activityLog}
        onClear={() => setActivityLog([])}
      />

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
            return next;
          });
        }}
        progress={visionPipelineProgress}
        cameraActive={cameraMode}
      />

      {infoModalOpen ? (
        <div
          className="modal info-modal"
          role="dialog"
          aria-modal="true"
          aria-labelledby="info-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) setInfoModalOpen(false);
          }}
        >
          <div className="modal-box">
            <h3 id="info-modal-title">טכנולוגיית GROVEE</h3>
            <p>GROVEE מריץ מודלי AI מתקדמים ישירות בדפדפן שלך באמצעות WebAssembly ו-Transformers.js.</p>
            <p>
              כל הנתונים נשארים במכשיר שלך. המודל <strong>GEMMA 4 E2B</strong> מורד פעם אחת (~3.9GB, כולל ראייה) ועובד
              במהירות שיא — גם בלי תלות בענן אחרי ההורדה.
            </p>
            <button type="button" className="close-modal" onClick={() => setInfoModalOpen(false)}>
              סגור
            </button>
          </div>
        </div>
      ) : null}

      {(phase === "start" || phase === "loading") && (
        <div
          id="intro-screen"
          className={`intro-screen ${phase === "loading" ? "intro-screen--loading" : ""}`}
          aria-busy={phase === "loading"}
          aria-live="polite"
        >
          <IntroCanvas />

          <div className="core-visual" aria-hidden="true">
            <div className="ring r1" />
            <div className="ring r2" />
            <div className="ring r3" />
          </div>

          <div className="intro-text">
            <div className="brand-title">GROVEE</div>
            <div className="model-name">GEMMA 4 E2B</div>

            {phase === "start" ? (
              <>
                <button
                  type="button"
                  className="load-btn"
                  onClick={loadModel}
                  disabled={isLoading || isGenerating || !!workerBootError}
                >
                  טען מודל מקומי
                </button>
                <button type="button" className="learn-link" onClick={() => setInfoModalOpen(true)}>
                  איך זה עובד?
                </button>
                <button
                  type="button"
                  className="learn-link learn-link--muted"
                  onClick={() => void clearModelCache()}
                  disabled={isGenerating || isLoading || cacheClearing}
                >
                  {cacheClearing ? "מנקה מטמון…" : "נקה מטמון"}
                </button>
              </>
            ) : (
              <>
                <div className="progress-wrapper progress-wrapper--visible">
                  <div
                    className="progress-fill"
                    role="progressbar"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.min(100, Math.round(progress))}
                    style={{ width: `${Math.min(100, progress)}%` }}
                  />
                </div>
                <div className="status-msg">{status}</div>
                {loadingPhase === "init" ? (
                  <p className="loading-file-detail">מאתחל ONNX — כמעט מוכן</p>
                ) : loadingByteLine ? (
                  <p className="loading-file-detail" dir="ltr">
                    {loadingByteLine}
                  </p>
                ) : null}
                <p className="loading-rotating-tip" key={loadingTipIndex} dir="rtl">
                  {loadingTip}
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {phase === "ready" && (
        <div
          id="app-container"
          className={`app-container app-container--visible ${artifactOpen ? "app-container--artifact-open" : ""} ${sidebarOpen ? "app-container--sidebar-open" : ""}`}
        >
          <div
            className={`sb-overlay ${sidebarOpen ? "active" : ""}`}
            onClick={() => setSidebarOpen(false)}
            aria-hidden="true"
          />

          <aside className={`sidebar ${sidebarOpen ? "active" : ""}`}>
            <div className="sb-header">
              <div className="sb-header-start">
                <div className="sb-logo">G</div>
                GROVEE
              </div>
              <button
                type="button"
                className="sb-close-btn"
                onClick={() => setSidebarOpen(false)}
                aria-label="סגור היסטוריה"
              >
                ×
              </button>
            </div>
            <button
              type="button"
              className="new-chat"
              onClick={() => {
                const id = newChatSessionId();
                setPendingAttachments((prev) => {
                  for (const p of prev) URL.revokeObjectURL(p.previewUrl);
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
                setSidebarOpen(false);
                setArtifactOpen(false);
              }}
              disabled={isGenerating}
            >
              צ&apos;אט חדש
            </button>
            <div className="history chat-list">
              {sortedSessions.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={`hist-item chat-item ${s.id === chatSessionsState.activeId ? "active" : ""}`}
                  onClick={() => {
                    if (s.id === chatSessionsState.activeId || isGenerating) return;
                    setChatSessionsState((st) => ({ ...st, activeId: s.id }));
                    setAssistantBuffer("");
                    assistantBufferRef.current = "";
                    setArtifactOpen(false);
                    setSidebarOpen(false);
                  }}
                  disabled={isGenerating}
                >
                  {s.title}
                </button>
              ))}
            </div>
            <div className="user-foot">
              <div className="avatar" aria-hidden="true" />
              <span>אורח</span>
              <button
                type="button"
                className="sb-settings-btn"
                title="הגדרות Gemma"
                aria-label="פתח הגדרות"
                onClick={() => {
                  setSettingsModalKey((k) => k + 1);
                  setSettingsOpen(true);
                }}
              >
                ⚙
              </button>
            </div>
          </aside>

          {artifactOpen && activeArtifact ? (
            <aside className="artifact-panel open" aria-label="חלונית קוד">
              <ArtifactPanel
                artifact={activeArtifact}
                streaming={isGenerating && !!assistantBuffer}
                onClose={() => setArtifactOpen(false)}
              />
            </aside>
          ) : null}

          <section className="chat-area">
            <header className="chat-header">
              <button
                type="button"
                className="sidebar-toggle"
                aria-label={sidebarOpen ? "סגור היסטוריה" : "פתח היסטוריה"}
                aria-expanded={sidebarOpen}
                onClick={() => setSidebarOpen((v) => !v)}
              >
                <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </svg>
              </button>
              <div className="model-badge">{modelLabel.toUpperCase()}</div>
              <div className="chat-header-actions">
                {activeArtifact && !artifactOpen ? (
                  <button
                    type="button"
                    className="artifact-reopen-btn"
                    onClick={() => setArtifactOpen(true)}
                  >
                    פתח {activeArtifact.kind === "html" ? "HTML" : "קוד"}
                  </button>
                ) : null}
                <button
                  type="button"
                  className="activity-log-btn vision-inspector-btn"
                  onClick={() => setVisionInspectorOpen(true)}
                  title="Vision Inspector — זיהוי חי, מודלים וזמני דגימה"
                  disabled={!cameraMode}
                >
                  🔬 Vision
                </button>
                <button
                  type="button"
                  className="activity-log-btn"
                  onClick={() => setActivityLogOpen(true)}
                  title="הצג את כל פעילות המודל — הנחיות, בקשות ותשובות"
                >
                  📋 פעילות
                  {activityLog.length ? (
                    <span className="activity-log-count">{activityLog.length}</span>
                  ) : null}
                </button>
              </div>
            </header>

            <div className="msg-list messages" ref={messagesListRef}>
              {messages.length === 0 && !assistantBuffer && (
                <div className="empty-state">
                  <div className="msg">
                    <div className="msg-icon ai">AI</div>
                    <div className="msg-txt">
                      היי! המודל Gemma 4 E2B מוכן לעבודה. מה אפשר לעזור לך ליצור היום?
                    </div>
                  </div>
                </div>
              )}
              {messages.map((msg) => (
                <article
                  key={msg.id}
                  className="msg"
                  dir={isRtlText(msg.content) ? "rtl" : "ltr"}
                >
                  <div className={`msg-icon ${msg.role === "user" ? "user" : "ai"}`}>
                    {msg.role === "user" ? "א" : "AI"}
                  </div>
                  <div className="msg-txt">
                    {msg.images?.length ? <UserAttachedImages images={msg.images} /> : null}
                    {msg.artifact ? (
                      <ArtifactChip
                        kind={msg.artifact.kind}
                        label={msg.artifact.kind === "html" ? "HTML" : msg.artifact.title}
                        onOpen={() => openArtifact(msg.artifact!)}
                      />
                    ) : null}
                    <MessageBody content={msg.content} onOpenArtifact={openArtifact} />
                  </div>
                </article>
              ))}
              {assistantBuffer && (
                <article className="msg">
                  <div className="msg-icon ai">AI</div>
                  <div className="msg-txt">
                    <MessageBody
                      content={assistantBuffer}
                      onOpenArtifact={openArtifact}
                      showThinking={thinkingMode}
                    />
                  </div>
                </article>
              )}
            </div>

            <div className="composer-modes">
              <label className="composer-mode-pill" title="מפעיל חשיבה native של Gemma 4 (<|think|>) — תהליך החשיבה יוצג לפני התשובה.">
                <input
                  type="checkbox"
                  checked={thinkingMode}
                  onChange={(e) => setThinkingMode(e.target.checked)}
                  disabled={isGenerating}
                />
                <span>Think</span>
              </label>
              <label
                className="composer-mode-pill"
                title="מושך קטעים מוויקיפדיה ומחיפוש מאגרים ב-GitHub."
              >
                <input
                  type="checkbox"
                  checked={webSearchMode}
                  onChange={(e) => setWebSearchMode(e.target.checked)}
                  disabled={isGenerating}
                />
                <span>Search</span>
              </label>
              <label
                className={`composer-mode-pill ${cameraMode ? "composer-mode-pill--active" : ""}`}
                title="מצב מצלמה חי — תצפית קלה (COCO+תנועה) עם Gemma מוגבל על מחשבים חלשים. מומלץ WASM בהגדרות על נייד."
              >
                <input
                  type="checkbox"
                  checked={cameraMode}
                  onChange={() => void toggleCameraMode()}
                  disabled={isGenerating || !isLoaded}
                />
                <span>🎥 Camera</span>
              </label>
              {cameraMode ? (
                <CameraPreview
                  ref={cameraVideoRef}
                  active={cameraMode}
                  observing={cameraObserving}
                  mood={characterMood}
                  error={cameraError}
                  visionResult={visionResult}
                  pipelineConfig={visionPipelineConfig}
                  pipelineProgress={visionPipelineProgress}
                  visionPaused={isGenerating}
                  onVideoReady={(video) => void startVisionPipeline(video)}
                  onPipelineConfigChange={(partial) => {
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
                  }}
                />
              ) : null}
              {cameraStatus ? (
                <span className="composer-status-hint composer-status-hint--camera">{cameraStatus}</span>
              ) : null}
              {status && status !== "Not loaded" && status !== "Ready" ? (
                <span className="composer-status-hint">{status}</span>
              ) : null}
            </div>

            <form
              className={`input-zone ${isDragOver ? "input-zone--drag" : ""}`}
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
                      <img src={p.previewUrl} alt="" className="composer-attachment-thumb" />
                      <button
                        type="button"
                        className="composer-attachment-remove"
                        onClick={() => removePendingAttachment(p.id)}
                        aria-label="הסר תמונה"
                        disabled={isGenerating}
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
                accept="image/jpeg,image/png,image/webp,image/gif"
                multiple
                hidden
                onChange={(e) => {
                  if (e.target.files?.length) void addFilesAsAttachments(e.target.files);
                  e.target.value = "";
                }}
              />
              <div className="in-box">
                <button
                  type="button"
                  className="in-act in-attach"
                  disabled={!isLoaded || isGenerating || pendingAttachments.length >= MAX_ATTACHMENTS}
                  aria-label="צרף תמונה"
                  title="צרף תמונה (או הדבק Ctrl+V)"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <path d="M21 15l-5-5L5 21" />
                  </svg>
                </button>
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
                  placeholder={pendingAttachments.length ? "שאל על התמונה…" : "הקלד הודעה או צרף תמונה…"}
                  rows={1}
                  disabled={!isLoaded || isGenerating}
                />
                {isGenerating ? (
                  <button
                    type="button"
                    className="in-act in-stop"
                    onClick={stopGeneration}
                    aria-label="עצור"
                    title="עצור יצירה"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  </button>
                ) : (
                  <button
                    type="submit"
                    className={`in-act in-send ${sendActive ? "in-send--active" : ""}`}
                    disabled={!isLoaded || !sendActive}
                    aria-label="שלח"
                    title="שלח"
                  >
                    <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <line x1="22" y1="2" x2="11" y2="13" />
                      <polygon points="22 2 15 22 11 13 2 9 22 2" />
                    </svg>
                  </button>
                )}
              </div>
            </form>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
