import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, FormEvent, ReactNode } from "react";
import {
  cleanEnglishImagePrompt,
  isCodeRequest,
  isImageGenerationRequest,
  isRtlText,
  isSimpleGreeting,
} from "./chatIntents";
import {
  generateSdTurboPng,
  getSdTurboSizeNote,
  revokeImageUrl,
  terminateLocalImageWorker,
} from "./localImageGen";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  modelLabel?: string;
  /** Inline preview for user messages sent with an image */
  imageDataUrl?: string;
  /**
   * Local SD-Turbo preview only (`blob:` / `data:`). Never use http(s) here — avoids broken offline links and markdown leaks.
   */
  localImageUrl?: string;
};

type WorkerOutMessage =
  | { type: "status"; text: string }
  | {
      type: "progress";
      text: string;
      progress: number;
      detail?: string;
      file?: string;
      loadMode?: "network" | "memory";
    }
  | { type: "loaded"; modelId: string; device: string }
  | { type: "caption_model_loaded"; modelId: string; device: string }
  | {
      type: "preload_all_done";
      textModels: number;
      captionModels: number;
      failedTextModelIds?: string[];
      failedCaptionModelIds?: string[];
    }
  | { type: "token"; text: string }
  | { type: "caption_done"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

const DEFAULT_MODEL = "onnx-community/gemma-4-E2B-it-ONNX";
/** Main chat models (Transformers.js ONNX). See https://huggingface.co/collections/webml-community/transformersjs-v4-demos */
const TEXT_CHAT_MODEL_OPTIONS = [
  { id: DEFAULT_MODEL, label: "Gemma 4 E2B — עברית/אנגלית (ברירת מחדל)" },
  {
    id: "LiquidAI/LFM2.5-1.2B-Thinking-ONNX",
    label: "LFM2.5 1.2B Thinking — מודל חשיבה (כבד, ~GB+)",
  },
  { id: "onnx-community/LFM2.5-350M-ONNX", label: "LFM2.5 350M — קל יחסית" },
] as const;

const KNOWN_TEXT_CHAT_MODEL_IDS = new Set<string>(TEXT_CHAT_MODEL_OPTIONS.map((o) => o.id));

const normalizeTextChatModelId = (id: string | undefined): string =>
  id && KNOWN_TEXT_CHAT_MODEL_IDS.has(id) ? id : DEFAULT_MODEL;

/** Public ONNX repo for Transformers.js (the *-Instruct-ONNX* repo returns 401 for anonymous fetch). */
const CODE_MODEL = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";
/** Set after Gemma finishes loading; used so cache-clear can reset UX hints. Vision/caption loads lazily on first image use so chat stays responsive. */
const MODEL_CACHE_FLAG = "grovee_models_warmed_v1";
const SETTINGS_STORAGE_KEY = "grovee_model_settings_v1";
const CHATS_STORAGE_KEY = "grovee_chats_v1";

/** Cache Storage bucket names used by deps; must stay in sync with web-txt2img / transformers.js defaults. */
const WEB_TXT2IMG_CACHE = "web-txt2img-v1";
const TRANSFORMERS_CACHE = "transformers-cache";

const shouldDeleteBrowserCache = (name: string): boolean => {
  const l = name.toLowerCase();
  return (
    l.includes("transformers") ||
    l.includes("huggingface") ||
    l.startsWith("hf-") ||
    l.includes("onnx") ||
    l.includes("ort-wasm") ||
    l.includes("ort.") ||
    l.includes("web-txt2img") ||
    l.includes("txt2img")
  );
};

/** When indexedDB.databases() is missing (some browsers), still drop known DB names. */
const INDEXEDDB_FALLBACK_NAMES = [TRANSFORMERS_CACHE, "hf-transformers-cache", "onnxruntime"];

type ChatSession = {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
};

type ChatSessionsState = { activeId: string; sessions: ChatSession[] };

const sessionTitleFromMessages = (sessionMessages: ChatMessage[]): string => {
  const firstUser = sessionMessages.find((m) => m.role === "user")?.content?.trim();
  if (!firstUser) return "שיחה חדשה";
  return firstUser.slice(0, 28) + (firstUser.length > 28 ? "…" : "");
};

const stripImageDataForStorage = (sessionMessages: ChatMessage[]): ChatMessage[] =>
  sessionMessages.map((m) => {
    const trimmed: ChatMessage = { id: m.id, role: m.role, content: m.content };
    if (m.modelLabel !== undefined) trimmed.modelLabel = m.modelLabel;
    return trimmed;
  });

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
  const serializable = {
    activeId: state.activeId,
    sessions: state.sessions.map((s) => ({
      ...s,
      messages: stripImageDataForStorage(s.messages),
    })),
  };
  try {
    localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(serializable));
  } catch {
    try {
      const trimmed = {
        activeId: state.activeId,
        sessions: state.sessions.map((s) => ({
          ...s,
          messages: stripImageDataForStorage(s.messages).map((m) => ({
            ...m,
            content: m.content.length > 12_000 ? `${m.content.slice(0, 12_000)}…` : m.content,
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

const shortLabelForTextModel = (id: string): string => {
  if (id.includes("LFM2.5") && id.toLowerCase().includes("thinking")) return "LFM2.5 Thinking";
  if (id.includes("LFM2.5")) return "LFM2.5";
  if (id.toLowerCase().includes("gemma")) return "Gemma 4";
  return "Assistant";
};

const VISION_MODEL_OPTIONS = [
  { id: "Xenova/vit-gpt2-image-captioning", label: "ViT-GPT2 Captioning (Fast)" },
  { id: "onnx-community/moondream2", label: "Moondream2 (Better detail)" },
] as const;

/** Chat + warmup only use the fast caption model; Moondream is omitted to avoid multi‑GB RAM / OOM on refresh. */
const DEFAULT_CAPTION_MODEL_ID = VISION_MODEL_OPTIONS[0].id;

type TunableModelSettings = {
  temperature: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  topP: number;
  systemPrompt: string;
};

const defaultGemmaSettings: TunableModelSettings = {
  temperature: 0.2,
  maxNewTokens: 512,
  repetitionPenalty: 1.12,
  topP: 0.9,
  systemPrompt:
    "You are a helpful assistant. Always respond in clear, well-formed sentences in the same language as the user (Hebrew stays RTL-friendly: full sentences, correct punctuation at end of sentence). Do not repeat role labels. When the user asks for HTML/CSS/JS (including a single-file page), output exactly one fenced block: ```html ... ``` containing a complete, valid document: <!DOCTYPE html>, <html lang=\"he\" dir=\"rtl\">, <head> with <meta charset=\"UTF-8\">, embedded <style> and <script> as needed, and <body>. No duplicate stray tags; no broken CSS.",
};

const defaultCoderSettings: TunableModelSettings = {
  temperature: 0.08,
  maxNewTokens: 768,
  repetitionPenalty: 1.06,
  topP: 0.88,
  systemPrompt:
    "You are an expert programmer. Output working code with brief comments. Prefer markdown code fences with language tags.",
};

/** ONNX Runtime backend: auto = WebGPU then WASM; wasm = CPU everywhere; webgpu = GPU only (fails on bad drivers). */
type InferenceBackendPreference = "auto" | "webgpu" | "wasm";

type AppSettings = {
  /** Primary ONNX text model loaded on Start (see TEXT_CHAT_MODEL_OPTIONS). */
  textChatModelId: string;
  /**
   * Optional Hugging Face Hub mirror (Transformers.js `env.remoteHost`), e.g. https://hf-mirror.com
   * when https://huggingface.co is blocked. Empty = official hub.
   */
  hfRemoteHost: string;
  /** Where Transformers.js runs models (important for different GPUs / drivers). */
  inferenceBackend: InferenceBackendPreference;
  gemma: TunableModelSettings;
  coder: TunableModelSettings;
  visionMaxTokens: number;
};

const defaultAppSettings = (): AppSettings => ({
  textChatModelId: DEFAULT_MODEL,
  hfRemoteHost: "",
  inferenceBackend: "auto",
  gemma: { ...defaultGemmaSettings },
  coder: { ...defaultCoderSettings },
  visionMaxTokens: 96,
});

const loadSettings = (): AppSettings => {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return defaultAppSettings();
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      ...defaultAppSettings(),
      textChatModelId: normalizeTextChatModelId(
        typeof parsed.textChatModelId === "string" ? parsed.textChatModelId : undefined,
      ),
      hfRemoteHost: typeof parsed.hfRemoteHost === "string" ? parsed.hfRemoteHost : "",
      inferenceBackend:
        parsed.inferenceBackend === "webgpu" || parsed.inferenceBackend === "wasm" || parsed.inferenceBackend === "auto"
          ? parsed.inferenceBackend
          : "auto",
      gemma: { ...defaultGemmaSettings, ...parsed.gemma },
      coder: { ...defaultCoderSettings, ...parsed.coder },
      visionMaxTokens:
        typeof parsed.visionMaxTokens === "number" && Number.isFinite(parsed.visionMaxTokens)
          ? Math.min(256, Math.max(32, Math.round(parsed.visionMaxTokens)))
          : defaultAppSettings().visionMaxTokens,
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

type ImageOrch = { step: "translate"; userText: string };

type CodeOrch =
  | { step: "route"; userText: string }
  | { step: "code"; userText: string; techTask: string }
  | { step: "summarize"; userText: string; techTask: string; codeOut: string };

type CaptionOrch = { step: "polish"; userText: string; rawCaption: string };

type Orchestration = { kind: "image"; data: ImageOrch } | { kind: "code"; data: CodeOrch } | { kind: "caption"; data: CaptionOrch };

/** Strip common chat-template junk from raw model output. */
const cleanModelOutput = (input: string): string => {
  const cleaned = input
    .replace(/\r/g, "")
    .split("\n")
    .filter((line) => !/^\s*(User|Assistant|System)\s*:/i.test(line))
    .join("\n")
    .replace(/^["']+|["']+$/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const lines = cleaned
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const deduped: string[] = [];
  for (const line of lines) {
    if (deduped[deduped.length - 1] !== line) deduped.push(line);
  }
  const normalized = deduped.join("\n");

  if (normalized.length) return normalized;
  const raw = input.trim();
  return raw.length ? raw.slice(0, 2000) : "No response generated.";
};

/** GitHub requires a User-Agent on API requests. */
const WEB_LOOKUP_USER_AGENT = "GROVEEMODEL/1.0 (browser chat; no backend)";

/**
 * When the user writes in Hebrew (no Latin tokens), map common tech phrases to English
 * so GitHub repository search can still return repos.
 */
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
    if (!response.ok) {
      console.warn("[GROVEE] Wikipedia HTTP", response.status, lang, endpoint.slice(0, 96));
      return "";
    }
    const data = (await response.json()) as [string, string[], string[], string[]];
    const titles = data[1] ?? [];
    const snippets = data[2] ?? [];
    const urls = data[3] ?? [];
    if (!titles.length) return "";
    return titles
      .map((title, i) => `- ${title}: ${snippets[i] ?? ""} (${urls[i] ?? ""})`)
      .join("\n");
  } catch (e) {
    console.warn("[GROVEE] Wikipedia fetch failed", lang, e);
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
    if (!response.ok) {
      console.warn("[GROVEE] GitHub API HTTP", response.status, url.slice(0, 120));
      return "";
    }
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
  } catch (e) {
    console.warn("[GROVEE] GitHub fetch failed", e);
    return "";
  }
};

/** Wikipedia (en + optional he) + GitHub repo search — not a full web search engine. */
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

type MsgPart = { type: "text" | "html" | "image" | "code"; value: string; lang?: string };

/** Build srcDoc for sandboxed iframe: full documents pass through; fragments get a minimal shell. */
const normalizeHtmlForIframe = (fragmentOrDoc: string): string => {
  const t = fragmentOrDoc.trim();
  const headSample = t.slice(0, 600).toLowerCase();
  if (headSample.includes("<!doctype") || headSample.startsWith("<html")) {
    if (!headSample.includes("charset")) {
      if (/<head\b/i.test(t)) {
        return t.replace(/<head\b[^>]*>/i, (h) => `${h}<meta charset="utf-8">`);
      }
      return `<!DOCTYPE html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${t}</body></html>`;
    }
    return t;
  }
  return `<!DOCTYPE html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{min-height:100%;margin:0;}</style></head><body>${t}</body></html>`;
};

const extractRichParts = (content: string): MsgPart[] => {
  const parts: MsgPart[] = [];
  let remaining = content;
  const pushText = (t: string) => {
    if (t) parts.push({ type: "text", value: t });
  };

  while (remaining.length) {
    type Cand = { idx: number; len: number; part: MsgPart };
    const candidates: Cand[] = [];

    const htmlFence = remaining.match(/```html\s*([\s\S]*?)(?:```|$)/i);
    if (htmlFence && htmlFence.index !== undefined && (htmlFence[1].trim().length > 0 || htmlFence[0].includes("```"))) {
      candidates.push({
        idx: htmlFence.index,
        len: htmlFence[0].length,
        part: { type: "html", value: htmlFence[1].trim() },
      });
    }

    const codeFence = remaining.match(/```(?!html)(\w*)\s*([\s\S]*?)```/i);
    if (codeFence && codeFence.index !== undefined) {
      candidates.push({
        idx: codeFence.index,
        len: codeFence[0].length,
        part: { type: "code", value: codeFence[2], lang: codeFence[1] || "text" },
      });
    }

    const imgMatch = remaining.match(/!\[([^\]]*)\]\(([^)]+)\)/);
    if (imgMatch && imgMatch.index !== undefined) {
      candidates.push({
        idx: imgMatch.index,
        len: imgMatch[0].length,
        part: { type: "image", value: imgMatch[2].trim() },
      });
    }

    const fullDoc = remaining.match(/(?:<!DOCTYPE\s+html[^>]*>|<html\b[^>]*>)[\s\S]*?<\/html>/i);
    if (fullDoc && fullDoc.index !== undefined) {
      candidates.push({
        idx: fullDoc.index,
        len: fullDoc[0].length,
        part: { type: "html", value: fullDoc[0].trim() },
      });
    }

    let best: Cand | null = null;
    for (const c of candidates) {
      if (!best || c.idx < best.idx) best = c;
    }

    if (!best) {
      pushText(remaining);
      break;
    }

    pushText(remaining.slice(0, best.idx));
    parts.push(best.part);
    remaining = remaining.slice(best.idx + best.len);
  }

  if (!parts.length) parts.push({ type: "text", value: content });
  return parts;
};

function HtmlSandboxBlock({ html }: { html: string }) {
  const [tab, setTab] = useState<"preview" | "source">("preview");
  const srcDoc = useMemo(() => normalizeHtmlForIframe(html), [html]);

  return (
    <div className="html-preview-wrap">
      <div className="html-preview-toolbar">
        <span className="html-preview-label">HTML</span>
        <div className="html-preview-tabs" role="tablist" aria-label="HTML view">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "preview"}
            className={`html-preview-tab ${tab === "preview" ? "active" : ""}`}
            onClick={() => setTab("preview")}
          >
            תצוגה חיה
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "source"}
            className={`html-preview-tab ${tab === "source" ? "active" : ""}`}
            onClick={() => setTab("source")}
          >
            מקור
          </button>
        </div>
      </div>
      {tab === "preview" ? (
        <iframe
          className="html-preview-frame"
          title="HTML preview"
          sandbox="allow-scripts allow-forms"
          srcDoc={srcDoc}
        />
      ) : (
        <pre className="html-source-block">
          <code>{html}</code>
        </pre>
      )}
    </div>
  );
}

function isLocalImageSrc(src: string): boolean {
  const s = src.trim();
  return s.startsWith("blob:") || s.startsWith("data:image");
}

function MessageBody({ content, localImageUrl }: { content: string; localImageUrl?: string }) {
  const parts = useMemo(() => extractRichParts(content), [content]);
  const dir = isRtlText(content) ? "rtl" : "ltr";
  const showLocal =
    typeof localImageUrl === "string" && localImageUrl.length > 0 && isLocalImageSrc(localImageUrl);

  return (
    <div className="msg-body" dir={dir}>
      {parts.map((part, i) => {
        if (part.type === "html" && part.value.length > 0) {
          return <HtmlSandboxBlock key={i} html={part.value} />;
        }
        if (part.type === "code") {
          return (
            <pre key={i} className="msg-code-block">
              <code className={part.lang ? `lang-${part.lang}` : undefined}>{part.value}</code>
            </pre>
          );
        }
        if (part.type === "image") {
          const raw = part.value.trim();
          if (!isLocalImageSrc(raw)) {
            return (
              <p key={i} className="msg-text msg-offsite-img-blocked">
                קישור תמונה מהרשת לא מוצג (רק תמונה מקומית מהדפדפן). אם עדיין מופיע קישור ישן — רענון קשיח או נקה מטמון
                האתר.
              </p>
            );
          }
          return (
            <div key={i} className="msg-image-wrap">
              <img className="msg-image" src={raw} alt="Generated locally" loading="lazy" />
            </div>
          );
        }
        return (
          <p key={i} className="msg-text">
            {part.value}
          </p>
        );
      })}
      {showLocal ? (
        <div className="msg-image-wrap">
          <img className="msg-image" src={localImageUrl} alt="Generated locally" loading="lazy" />
        </div>
      ) : null}
    </div>
  );
}

function SettingsModal({
  open,
  onClose,
  settings,
  onSave,
}: {
  open: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSave: (s: AppSettings) => void;
}) {
  const [draft, setDraft] = useState<AppSettings>(() => settings);

  if (!open) return null;

  const row = (label: string, children: ReactNode) => (
    <label className="settings-row">
      <span>{label}</span>
      {children}
    </label>
  );

  return (
    <div className="settings-overlay" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div className="settings-panel glass">
        <div className="settings-head">
          <h2 id="settings-title">Model settings</h2>
          <button type="button" className="icon-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <p className="settings-hint">
          מודל הצ&apos;אט הראשי (למשל Gemma או LFM Thinking) מטפל בשיחה; לבקשות קוד האפליקציה מפיקה משימה טכנית באנגלית ואז{" "}
          <strong>Qwen Coder</strong> כותב קוד. תמונות: פרומפט באנגלית אוטומטית, ענן או SD-Turbo מקומי.
        </p>

        <section className="settings-section">
          <h3>מודל צ&apos;אט ראשי (ONNX / Transformers.js)</h3>
          <p className="settings-micro">
            אוסף דמוים:{" "}
            <a
              href="https://huggingface.co/collections/webml-community/transformersjs-v4-demos"
              target="_blank"
              rel="noreferrer"
            >
              Transformers.js V4 demos
            </a>
            . אחרי שינוי מודל יש ללחוץ שוב <strong>התחל</strong> כדי לטעון משקולות.
          </p>
          <label className="settings-row block">
            <span>בחירת מודל</span>
            <select
              value={draft.textChatModelId}
              onChange={(e) => setDraft((d) => ({ ...d, textChatModelId: e.target.value }))}
            >
              {TEXT_CHAT_MODEL_OPTIONS.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="settings-row block">
            <span>HF mirror (אם Failed to fetch)</span>
            <input
              type="url"
              inputMode="url"
              autoComplete="off"
              spellCheck={false}
              placeholder="ריק = huggingface.co · לדוגמה: https://hf-mirror.com"
              value={draft.hfRemoteHost}
              onChange={(e) => setDraft((d) => ({ ...d, hfRemoteHost: e.target.value }))}
            />
          </label>
          <p className="settings-micro">
            אם יש <code>Failed to fetch</code> ל־huggingface.co, האפליקציה מנסה פעם אחת אוטומטית דרך hf-mirror.com. אם גם זה נכשל: הגדר מראה ידנית, שמור → נקה מטמון מודל → התחל (או רשת/VPN אחרים).
          </p>
          <label className="settings-row block">
            <span>מנוע חישוב (כל כרטיסי המסך)</span>
            <select
              value={draft.inferenceBackend}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  inferenceBackend: e.target.value as InferenceBackendPreference,
                }))
              }
            >
              <option value="auto">Auto — WebGPU אם אפשר, אחרת WASM על המעבד</option>
              <option value="wasm">WASM — רק מעבד (יציב בכל מחשב, איטי יותר)</option>
              <option value="webgpu">WebGPU — רק כרטיס מתאים (כשלים אפשריים אם אין תמיכה)</option>
            </select>
          </label>
          <p className="settings-micro">
            כרטיסים ודרייברים שונים מתנהגים אחרת; Auto מומלץ. אחרי שינוי: שמור → לחץ שוב <strong>התחל</strong>.
          </p>
          <p className="settings-micro">
            Linux (מינט וכו&apos;): אם אין מתאם WebGPU, האפליקציה עוברת אוטומטית ל־WASM על המעבד — השיחה אמורה לעלות. ל־GPU:
            דפדפן מעודכן (Chromium מומלץ), דרייברים ל־NVIDIA / AMD / Intel, ובלינוקס לעיתים חבילות Vulkan (למשל{" "}
            <code>mesa-vulkan-drivers</code>).
          </p>
        </section>

        <section className="settings-section">
          <h3>פרמטרי צ&apos;אט (טמפרטורה, פרומפט מערכת)</h3>
          {row(
            "Temperature",
            <input
              type="number"
              step={0.05}
              min={0}
              max={2}
              value={draft.gemma.temperature}
              onChange={(e) =>
                setDraft((d) => ({ ...d, gemma: { ...d.gemma, temperature: Number(e.target.value) } }))
              }
            />,
          )}
          {row(
            "Max tokens",
            <input
              type="number"
              min={32}
              max={2048}
              value={draft.gemma.maxNewTokens}
              onChange={(e) =>
                setDraft((d) => ({ ...d, gemma: { ...d.gemma, maxNewTokens: Number(e.target.value) } }))
              }
            />,
          )}
          {row(
            "Top P",
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={draft.gemma.topP}
              onChange={(e) => setDraft((d) => ({ ...d, gemma: { ...d.gemma, topP: Number(e.target.value) } }))}
            />,
          )}
          {row(
            "Repetition penalty",
            <input
              type="number"
              step={0.02}
              min={1}
              max={2}
              value={draft.gemma.repetitionPenalty}
              onChange={(e) =>
                setDraft((d) => ({ ...d, gemma: { ...d.gemma, repetitionPenalty: Number(e.target.value) } }))
              }
            />,
          )}
          <label className="settings-row block">
            <span>System prompt</span>
            <textarea
              rows={3}
              value={draft.gemma.systemPrompt}
              onChange={(e) => setDraft((d) => ({ ...d, gemma: { ...d.gemma, systemPrompt: e.target.value } }))}
            />
          </label>
        </section>

        <section className="settings-section">
          <h3>Code model (Qwen Coder)</h3>
          {row(
            "Temperature",
            <input
              type="number"
              step={0.05}
              min={0}
              max={1.5}
              value={draft.coder.temperature}
              onChange={(e) =>
                setDraft((d) => ({ ...d, coder: { ...d.coder, temperature: Number(e.target.value) } }))
              }
            />,
          )}
          {row(
            "Max tokens",
            <input
              type="number"
              min={64}
              max={2048}
              value={draft.coder.maxNewTokens}
              onChange={(e) =>
                setDraft((d) => ({ ...d, coder: { ...d.coder, maxNewTokens: Number(e.target.value) } }))
              }
            />,
          )}
          {row(
            "Top P",
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={draft.coder.topP}
              onChange={(e) => setDraft((d) => ({ ...d, coder: { ...d.coder, topP: Number(e.target.value) } }))}
            />,
          )}
          <label className="settings-row block">
            <span>System prompt</span>
            <textarea
              rows={2}
              value={draft.coder.systemPrompt}
              onChange={(e) => setDraft((d) => ({ ...d, coder: { ...d.coder, systemPrompt: e.target.value } }))}
            />
          </label>
        </section>

        <section className="settings-section">
          <h3>Vision caption</h3>
          {row(
            "Max caption tokens",
            <input
              type="number"
              min={32}
              max={256}
              value={draft.visionMaxTokens}
              onChange={(e) => setDraft((d) => ({ ...d, visionMaxTokens: Number(e.target.value) }))}
            />,
          )}
        </section>

        <section className="settings-section">
          <h3>Image generation</h3>
          <p className="settings-micro">
            תמונות נוצרות רק מקומית בדפדפן (SD-Turbo דרך web-txt2img) — בלי לינקי HTTP וללא שרת תמונה חיצוני. פעם ראשונה
            דורשת רשת כדי למשוך משקולות (~2.3GB); אחרי מכן נשמרות במטמון הדפדפן וניתן לעבוד offline. {getSdTurboSizeNote()}
          </p>
          <p className="settings-micro">
            See{" "}
            <a href="https://github.com/lacerbi/web-txt2img" target="_blank" rel="noreferrer">
              web-txt2img
            </a>
            .
          </p>
        </section>

        <div className="settings-actions">
          <button type="button" className="subtle-btn" onClick={() => setDraft(defaultAppSettings())}>
            Reset defaults
          </button>
          <button
            type="button"
            className="pill-button"
            onClick={() => {
              onSave(draft);
              onClose();
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const workerRef = useRef<Worker | null>(null);
  const assistantBufferRef = useRef("");
  const activeModelShortLabelRef = useRef("Assistant");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const orchRef = useRef<Orchestration | null>(null);

  const [modelId, setModelId] = useState(() => normalizeTextChatModelId(loadSettings().textChatModelId));
  const [appSettings, setAppSettings] = useState<AppSettings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsModalKey, setSettingsModalKey] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState("Not loaded");
  const [progress, setProgress] = useState(0);
  /** Technical line from worker (bytes / file); cleared when load completes. */
  const [loadingDetail, setLoadingDetail] = useState("");
  /** Worker/script/load failures — visible banner + Console already logged. */
  const [workerBootError, setWorkerBootError] = useState<string | null>(null);
  /** Long-running load hints (weak hardware / network). */
  const [loadingSlowHint, setLoadingSlowHint] = useState("");
  const [prompt, setPrompt] = useState("");
  const [chatSessionsState, setChatSessionsState] = useState<ChatSessionsState>(() => loadChatSessionsState());
  const [assistantBuffer, setAssistantBuffer] = useState("");
  const [thinkingMode, setThinkingMode] = useState(false);
  const [webSearchMode, setWebSearchMode] = useState(false);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  const lastGeneratedUrlRef = useRef<string | null>(null);
  const [composerDragOver, setComposerDragOver] = useState(false);
  /** Bumping recreates the model worker so cache clear + stuck loads actually reset runtime state. */
  const [workerGeneration, setWorkerGeneration] = useState(0);
  const appSettingsRef = useRef(appSettings);
  const thinkingRef = useRef(thinkingMode);
  const webSearchRef = useRef(webSearchMode);
  const isLoadingRef = useRef(isLoading);

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
    isLoadingRef.current = isLoading;
  }, [isLoading]);

  useEffect(() => {
    saveChatSessionsState(chatSessionsState);
  }, [chatSessionsState]);

  const activeSession = useMemo(
    () =>
      chatSessionsState.sessions.find((s) => s.id === chatSessionsState.activeId) ?? chatSessionsState.sessions[0],
    [chatSessionsState],
  );
  const messages = activeSession.messages;
  const conversationTitle = activeSession.title;

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

  useEffect(() => {
    const prev = lastGeneratedUrlRef.current;
    if (prev && prev !== generatedImageUrl && prev.startsWith("blob:")) {
      revokeImageUrl(prev);
    }
    lastGeneratedUrlRef.current = generatedImageUrl;
  }, [generatedImageUrl]);

  const phase = isLoaded ? "ready" : isLoading ? "loading" : "start";
  const modelLabel = useMemo(() => modelId.split("/").pop() ?? modelId, [modelId]);

  useEffect(() => {
    if (phase !== "loading" || !isLoading) {
      queueMicrotask(() => setLoadingSlowHint(""));
      return;
    }
    const t2m = window.setTimeout(() => {
      setLoadingSlowHint(
        "עדיין טוען… על חומרה חלשה או רשת איטית זה נורמלי. נסה בהגדרות: מנוע WASM או HF mirror.",
      );
    }, 120_000);
    const t10m = window.setTimeout(() => {
      setLoadingSlowHint(
        "מעל 10 דקות: פתח F12 → Network/Console. נסה נקה מטמון מודל, רשת אחרת, או WASM.",
      );
    }, 600_000);
    return () => {
      clearTimeout(t2m);
      clearTimeout(t10m);
    };
  }, [phase, isLoading]);

  useEffect(() => {
    activeModelShortLabelRef.current = shortLabelForTextModel(modelId);
  }, [modelId]);

  const placeholder = useMemo(() => {
    if (!isLoaded) return "Start the app first…";
    return "Ask anything…";
  }, [isLoaded]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el || phase !== "ready") return;
    el.style.height = "auto";
    el.style.height = `${Math.min(Math.max(el.scrollHeight, 44), 160)}px`;
  }, [prompt, phase, imageDataUrl]);

  const ingestImageFile = useCallback((file: File | undefined) => {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => setImageDataUrl(String(reader.result));
    reader.readAsDataURL(file);
  }, []);

  const onComposerDragEnter = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.dataTransfer.types.includes("Files")) return;
    setComposerDragOver(true);
  }, []);

  const onComposerDragLeave = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const related = e.relatedTarget as Node | null;
    if (related && e.currentTarget.contains(related)) return;
    setComposerDragOver(false);
  }, []);

  const onComposerDragOver = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const onComposerDrop = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setComposerDragOver(false);
      ingestImageFile(e.dataTransfer.files?.[0]);
    },
    [ingestImageFile],
  );

  const runGemmaGenerate = (
    promptText: string,
    systemPrompt: string,
    maxNewTokens: number,
    temperature: number,
    repetitionPenalty: number,
    topP: number,
    webContext: string,
  ) => {
    workerRef.current?.postMessage({
      type: "generate",
      modelId: normalizeTextChatModelId(appSettingsRef.current.textChatModelId),
      prompt: promptText,
      systemPrompt,
      maxNewTokens,
      temperature,
      repetitionPenalty,
      topP,
      thinkingMode: thinkingRef.current,
      webContext,
    });
  };

  const runCoderGenerate = (promptText: string, webContext: string) => {
    const c = appSettingsRef.current.coder;
    workerRef.current?.postMessage({
      type: "generate",
      modelId: CODE_MODEL,
      prompt: promptText,
      systemPrompt: c.systemPrompt,
      maxNewTokens: c.maxNewTokens,
      temperature: c.temperature,
      repetitionPenalty: c.repetitionPenalty,
      topP: c.topP,
      thinkingMode: false,
      webContext,
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
        ev.message ||
          `Worker script failed (404/CORS?). File: ${ev.filename ?? "model.worker"}`,
      );
      setIsLoading(false);
    };

    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;
      if (msg.type === "status") {
        setStatus(msg.text);
      } else if (msg.type === "progress") {
        const trackProgressUi = isLoadingRef.current;
        if (trackProgressUi) {
          setStatus(msg.text);
          setLoadingDetail(msg.detail ?? "");
          const next = Math.min(100, Math.max(0, Math.round(Number(msg.progress) || 0)));
          setProgress(next);
        }
      } else if (msg.type === "loaded") {
        setWorkerBootError(null);
        setIsLoaded(true);
        setIsLoading(false);
        setProgress(100);
        setLoadingDetail("");
        setStatus(`מוכן לצ'אט — ${formatInferenceDevice(msg.device)} · מודל תמונה/כיתוב ייטען ברקע רק כשצריך`);
        try {
          localStorage.setItem(MODEL_CACHE_FLAG, "1");
        } catch {
          // ignore
        }
      } else if (msg.type === "caption_model_loaded") {
        setStatus(`Vision model ready on ${formatInferenceDevice(msg.device)}`);
      } else if (msg.type === "preload_all_done") {
        setWorkerBootError(null);
        setIsLoading(false);
        setIsLoaded(true);
        try {
          localStorage.setItem(MODEL_CACHE_FLAG, "1");
        } catch {
          // ignore
        }
        setProgress(100);
        setLoadingDetail("");
        const failedT = msg.failedTextModelIds?.length
          ? ` · נכשלו מודלי טקסט: ${msg.failedTextModelIds.join(", ")}`
          : "";
        const failedC = msg.failedCaptionModelIds?.length
          ? ` · נכשל vision: ${msg.failedCaptionModelIds.join(", ")}`
          : "";
        setStatus(
          failedT || failedC
            ? `מוכן חלקית: ${msg.textModels} טקסט + ${msg.captionModels} vision${failedT}${failedC}`
            : `מוכנים: Gemma + כיתוב תמונה (${msg.captionModels}) — קוד בטעינה עצלה`,
        );
      } else if (msg.type === "token") {
        setAssistantBuffer((prev) => {
          const next = prev + msg.text;
          assistantBufferRef.current = next;
          return next;
        });
      } else if (msg.type === "caption_done") {
        const raw = msg.text.trim();
        if (!raw) {
          setIsGenerating(false);
          return;
        }
        orchRef.current = {
          kind: "caption",
          data: { step: "polish", userText: "", rawCaption: raw },
        };
        setAssistantBuffer("");
        assistantBufferRef.current = "";
        setStatus("Gemma: formatting caption in your language…");
        const g = appSettingsRef.current.gemma;
        runGemmaGenerate(
          `Raw image description (English):\n${raw}\n\nRewrite for the user: one short paragraph in their language if the chat is Hebrew use Hebrew; keep meaning; no preamble.`,
          g.systemPrompt,
          Math.min(256, g.maxNewTokens),
          Math.min(0.25, g.temperature),
          g.repetitionPenalty,
          g.topP,
          "",
        );
      } else if (msg.type === "done") {
        const orch = orchRef.current;
        const buf = cleanModelOutput(assistantBufferRef.current);

        if (orch?.kind === "caption" && orch.data.step === "polish") {
          const polished = buf || orch.data.rawCaption;
          orchRef.current = null;
          setIsGenerating(false);
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: polished,
              modelLabel: activeModelShortLabelRef.current,
            },
          ]);
          setAssistantBuffer("");
          assistantBufferRef.current = "";
          return;
        }

        if (orch?.kind === "image" && orch.data.step === "translate") {
          const english = cleanEnglishImagePrompt(assistantBufferRef.current);
          const userText = orch.data.userText;
          assistantBufferRef.current = "";
          setAssistantBuffer("");

          void (async () => {
            const local = await generateSdTurboPng(english, (s) => setStatus(s));
            if (local.ok) {
              const imageUrl = local.objectUrl;
              setGeneratedImageUrl(imageUrl);
              orchRef.current = null;
              setIsGenerating(false);
              const caption = isRtlText(userText)
                ? `התמונה נוצרה מקומית בדפדפן בלבד (SD-Turbo). בלי שרת ענן ובלי קישורי HTTP — רק Blob מקומי.\nפרומפט באנגלית: ${english}`
                : `Image generated locally in your browser only (SD-Turbo). No cloud and no HTTP URLs — local blob only.\nEnglish prompt: ${english}`;
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: caption,
                  localImageUrl: imageUrl,
                  modelLabel: activeModelShortLabelRef.current,
                },
              ]);
              setStatus(isRtlText(userText) ? "מוכן" : "Ready");
              return;
            }
            orchRef.current = null;
            setIsGenerating(false);
            setAssistantBuffer("");
            assistantBufferRef.current = "";
            setStatus("Local image failed");
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: `יצירת התמונה המקומית נכשלה: ${local.message}. עם רשת — טען את האפליקציה פעם אחת כדי שמשקולות SD-Turbo יישמרו במטמון הדפדפן, ואז אפשר לעבוד offline.`,
                modelLabel: activeModelShortLabelRef.current,
              },
            ]);
          })();
          return;
        }

        if (orch?.kind === "code" && orch.data.step === "route") {
          const tech = buf || orch.data.userText;
          orchRef.current = { kind: "code", data: { step: "code", userText: orch.data.userText, techTask: tech } };
          assistantBufferRef.current = "";
          setAssistantBuffer("");
          setStatus("Code model running…");
          void (async () => {
            const techTask = tech;
            let webContext = "";
            if (webSearchRef.current) {
              try {
                webContext = await fetchWebContext(techTask);
              } catch {
                webContext = "";
              }
            }
            runCoderGenerate(techTask, webContext);
          })();
          return;
        }

        if (orch?.kind === "code" && orch.data.step === "code") {
          const codeOut = assistantBufferRef.current.trim() || buf;
          orchRef.current = {
            kind: "code",
            data: {
              step: "summarize",
              userText: orch.data.userText,
              techTask: orch.data.techTask,
              codeOut,
            },
          };
          assistantBufferRef.current = "";
          setAssistantBuffer("");
          setStatus("Gemma: summarizing in your language…");
          const g = appSettingsRef.current.gemma;
          const userLang = isRtlText(orch.data.userText) ? "Hebrew" : "the user's language";
          runGemmaGenerate(
            `Original user request:\n${orch.data.userText}\n\nTechnical task (English):\n${orch.data.techTask}\n\nCode model output:\n${codeOut}\n\nSummarize in ${userLang} what was done. Then include the full code in a markdown fenced block with the right language tag.`,
            g.systemPrompt,
            Math.min(700, g.maxNewTokens),
            g.temperature,
            g.repetitionPenalty,
            g.topP,
            "",
          );
          return;
        }

        if (orch?.kind === "code" && orch.data.step === "summarize") {
          const finalText = buf;
          orchRef.current = null;
          setIsGenerating(false);
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: finalText,
              modelLabel: activeModelShortLabelRef.current,
            },
          ]);
          setAssistantBuffer("");
          assistantBufferRef.current = "";
          return;
        }

        setIsGenerating(false);
        let output = buf;
        if (output.length > 4000) output = `${output.slice(0, 4000).trimEnd()}…`;
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: output,
            modelLabel: activeModelShortLabelRef.current,
          },
        ]);
        setAssistantBuffer("");
        assistantBufferRef.current = "";
      } else if (msg.type === "error") {
        orchRef.current = null;
        setIsGenerating(false);
        setIsLoading(false);
        setProgress(0);
        setLoadingDetail("");
        setStatus(`Error: ${msg.error}`);
        setWorkerBootError(msg.error);
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
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [setMessages, workerGeneration]);

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
    const mid = normalizeTextChatModelId(appSettingsRef.current.textChatModelId);
    setModelId(mid);
    setWorkerBootError(null);
    setLoadingSlowHint("");
    setIsLoading(true);
    setStatus(`Loading ${mid.split("/").pop() ?? mid}…`);
    setProgress(0);
    setLoadingDetail("");
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
      modelId: mid,
      dtype: "q4",
    });
  };

  const clearModelCache = async () => {
    if (isGenerating) {
      setStatus("המתן לסיום התשובה — לא ניתן לנקות מטמון בזמן יצירה.");
      return;
    }
    setStatus("מנקה מטמון מודלים…");
    setIsLoading(false);
    setProgress(0);
    setLoadingDetail("");
    setIsLoaded(false);
    setWorkerBootError(null);
    terminateLocalImageWorker();
    revokeImageUrl(generatedImageUrl);
    setGeneratedImageUrl(null);
    lastGeneratedUrlRef.current = null;
    localStorage.removeItem(MODEL_CACHE_FLAG);
    orchRef.current = null;
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setWorkerGeneration((g) => g + 1);

    try {
      if ("caches" in window) {
        const cacheKeys = await caches.keys();
        const toDelete = new Set<string>();
        for (const key of cacheKeys) {
          if (shouldDeleteBrowserCache(key)) toDelete.add(key);
        }
        toDelete.add(WEB_TXT2IMG_CACHE);
        toDelete.add(TRANSFORMERS_CACHE);
        await Promise.all([...toDelete].map((key) => caches.delete(key)));
      }

      const indexedDbGlobal = indexedDB as IDBFactory & {
        databases?: () => Promise<Array<{ name?: string }>>;
      };
      const deleteIdb = (name: string) => {
        try {
          indexedDB.deleteDatabase(name);
        } catch {
          /* ignore */
        }
      };
      for (const name of INDEXEDDB_FALLBACK_NAMES) {
        deleteIdb(name);
      }
      if (indexedDbGlobal.databases) {
        const dbs = await indexedDbGlobal.databases();
        for (const db of dbs) {
          const name = db.name ?? "";
          if (!name) continue;
          const lower = name.toLowerCase();
          if (
            lower.includes("transformers") ||
            lower.includes("huggingface") ||
            lower.includes("onnx") ||
            lower.includes("web-txt2img") ||
            lower.includes("txt2img")
          ) {
            deleteIdb(name);
          }
        }
      }

      setStatus("המטמון נוקה. לחץ «התחל» כדי לטעון מחדש.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`ניקוי מטמון נכשל: ${message}`);
    }
  };

  const persistSettings = (s: AppSettings) => {
    const normalized: AppSettings = {
      ...s,
      textChatModelId: normalizeTextChatModelId(s.textChatModelId),
    };
    setAppSettings((prev) => {
      const modelChanged = normalized.textChatModelId !== prev.textChatModelId;
      const backendChanged = normalized.inferenceBackend !== prev.inferenceBackend;
      if (modelChanged || backendChanged) {
        queueMicrotask(() => {
          if (modelChanged) setModelId(normalized.textChatModelId);
          setIsLoaded(false);
          setIsLoading(false);
          setStatus(
            modelChanged && backendChanged
              ? "מודל או מנוע חישוב השתנו — לחץ «התחל» כדי לטעון מחדש"
              : backendChanged
                ? "מנוע חישוב השתנה — לחץ «התחל» כדי לטעון מחדש"
                : "מודל צ'אט השתנה — לחץ «התחל» כדי לטעון מחדש",
          );
        });
      }
      return normalized;
    });
    saveSettings(normalized);
  };

  const sendPrompt = async (e: FormEvent) => {
    e.preventDefault();
    if (!workerRef.current || !isLoaded || isGenerating) return;
    const trimmed = prompt.trim();
    if (!trimmed && !imageDataUrl) return;

    if (imageDataUrl) {
      const captionModelId = DEFAULT_CAPTION_MODEL_ID;
      const userLine = trimmed || "תאר את התמונה";
      const attachment = imageDataUrl;
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: userLine, imageDataUrl: attachment },
      ]);
      setPrompt("");
      setImageDataUrl(null);
      setIsGenerating(true);
      orchRef.current = null;
      workerRef.current.postMessage({
        type: "caption",
        imageDataUrl: attachment,
        prompt: trimmed || undefined,
        modelId: captionModelId,
        maxNewTokens: appSettings.visionMaxTokens,
      });
      return;
    }

    if (!trimmed) return;

    if (isImageGenerationRequest(trimmed)) {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);
      setPrompt("");
      setAssistantBuffer("");
      assistantBufferRef.current = "";
      setIsGenerating(true);
      orchRef.current = { kind: "image", data: { step: "translate", userText: trimmed } };
      setStatus("Gemma: preparing English image prompt…");
      runGemmaGenerate(
        `User message (may be Hebrew):\n${trimmed}\n\nOutput ONLY a single English image-generation prompt, max 40 words, no quotes, no explanation.`,
        "You output only the English prompt text for an image model. No other text.",
        120,
        0.1,
        1.05,
        0.85,
        "",
      );
      return;
    }

    if (isCodeRequest(trimmed)) {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);
      setPrompt("");
      setAssistantBuffer("");
      assistantBufferRef.current = "";
      setIsGenerating(true);
      orchRef.current = { kind: "code", data: { step: "route", userText: trimmed } };
      setStatus("Gemma: routing to code model…");
      runGemmaGenerate(
        `User message (may be Hebrew):\n${trimmed}\n\nOutput ONLY a concise technical coding task in English (one short paragraph). No preamble, no labels.`,
        "You extract a coding task for a code LLM. English only in the answer body.",
        200,
        0.05,
        1.05,
        0.85,
        "",
      );
      return;
    }

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);
    setPrompt("");
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setIsGenerating(true);
    orchRef.current = null;

    const g = appSettings.gemma;
    const greeting = isSimpleGreeting(trimmed);
    let webContext = "";
    let searchHint = "";
    if (webSearchMode) {
      setStatus("Searching…");
      try {
        webContext = await fetchWebContext(trimmed);
        if (!webContext.trim()) searchHint = " · אין תוצאות חיפוש — מענה בלי הקשר רשת";
      } catch (e) {
        console.warn("[GROVEE] fetchWebContext failed:", e);
        webContext = "";
        searchHint = " · חיפוש נכשל — מענה בלי הקשר רשת";
      }
    }
    setStatus(`Generating…${searchHint}`);

    runGemmaGenerate(
      trimmed,
      greeting
        ? `${g.systemPrompt} If the user sends only a greeting, reply with one short friendly sentence only.`
        : g.systemPrompt,
      greeting ? 40 : g.maxNewTokens,
      greeting ? 0 : g.temperature,
      g.repetitionPenalty,
      g.topP,
      webContext,
    );
  };

  return (
    <main className="app theme-space">
      <div className="bg-overlay" />
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
      />

      {phase === "start" && (
        <section className="hero-screen glass hero-minimal">
          <h1 className="hero-brand">GROVEE</h1>
          <p className="hero-tagline">צ&apos;אט פרטי בדפדפן · עברית ואנגלית · מודלים נטענים אצלך במחשב</p>
          <div className="hero-actions">
            <button className="pill-button" onClick={loadModel} disabled={isLoading || isGenerating}>
              התחל
            </button>
            <button className="pill-button subtle-btn" onClick={() => void clearModelCache()} disabled={isGenerating}>
              נקה מטמון
            </button>
          </div>
          <p className="hero-status" aria-live="polite">
            {status}
          </p>
        </section>
      )}

      {phase === "loading" && isLoading && (
        <section className="loading-screen glass" aria-busy="true" aria-live="polite">
          <h2>GROVEE</h2>
          <div className="loading-model">{modelLabel}</div>
          <p className="loading-headline">
            <span className="loading-headline-status" title={status}>
              {status}
            </span>
            <span className="loading-headline-pct" aria-hidden="true">
              {Math.min(100, Math.round(progress))}%
            </span>
          </p>
          <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
            <div className="meter-fill" style={{ width: `${Math.min(100, progress)}%` }} />
          </div>
          <p className="loading-subline" title={loadingDetail || undefined}>
            {loadingDetail || "\u00a0"}
          </p>
          {loadingSlowHint ? <p className="loading-slow-hint">{loadingSlowHint}</p> : null}
          <p className="loading-meter-note">
            מד התקדמות אחד: הורדה מהרשת או קריאה ממטמון הדפדפן — עד 100% כשהמודל באמת מוכן.
          </p>
          <div className="hero-actions" style={{ marginTop: 16 }}>
            <button
              type="button"
              className="pill-button subtle-btn"
              onClick={() => void clearModelCache()}
              disabled={isGenerating}
            >
              נקה מטמון ואיפוס טעינה
            </button>
          </div>
        </section>
      )}

      {phase === "ready" && (
        <section className="ready-shell">
          <aside className="sidebar">
            <h3>GROVEE</h3>
            <button
              type="button"
              className="new-chat-btn"
              onClick={() => {
                const id = newChatSessionId();
                setChatSessionsState((s) => ({
                  activeId: id,
                  sessions: [{ id, title: "שיחה חדשה", updatedAt: Date.now(), messages: [] }, ...s.sessions],
                }));
                setAssistantBuffer("");
                assistantBufferRef.current = "";
                setPrompt("");
                setImageDataUrl(null);
                setComposerDragOver(false);
                revokeImageUrl(generatedImageUrl);
                setGeneratedImageUrl(null);
                lastGeneratedUrlRef.current = null;
                orchRef.current = null;
              }}
              disabled={isGenerating}
            >
              + New chat
            </button>
            <div className="chat-list">
              {sortedSessions.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={`chat-item ${s.id === chatSessionsState.activeId ? "active" : ""}`}
                  onClick={() => {
                    if (s.id === chatSessionsState.activeId || isGenerating) return;
                    setChatSessionsState((st) => ({ ...st, activeId: s.id }));
                    setAssistantBuffer("");
                    assistantBufferRef.current = "";
                    orchRef.current = null;
                  }}
                  disabled={isGenerating}
                >
                  {s.title}
                </button>
              ))}
            </div>
          </aside>

          <section className="chat-panel">
            <header className="chat-header">
              <div className="chat-header-row">
                <div>
                  <h2>{conversationTitle}</h2>
                  <p className="top-status">{status}</p>
                </div>
                <div className="header-actions">
                  <button
                    type="button"
                    className="icon-gear"
                    title="Model settings"
                    aria-label="Open settings"
                    onClick={() => {
                      setSettingsModalKey((k) => k + 1);
                      setSettingsOpen(true);
                    }}
                  >
                    ⚙
                  </button>
                </div>
              </div>
            </header>

            <div className="messages">
              {messages.length === 0 && !assistantBuffer && (
                <div className="empty-state">
                  <h3>שיחה חדשה</h3>
                  <p>
                    כתוב בעברית או באנגלית. צרף תמונה לתיאור. בקש קוד או &quot;צור תמונה&quot; — המערכת תבחר את המודל המתאים ותחזיר תשובה אחת ברורה.
                  </p>
                </div>
              )}
              {messages.map((msg) => (
                <article key={msg.id} className={`bubble ${msg.role}`} dir={isRtlText(msg.content) ? "rtl" : "ltr"}>
                  <strong>{msg.role === "user" ? "You" : msg.modelLabel ?? "Assistant"}</strong>
                  {msg.role === "user" && msg.imageDataUrl ? (
                    <div className="bubble-user-image">
                      <img src={msg.imageDataUrl} alt="" />
                    </div>
                  ) : null}
                  <MessageBody content={msg.content} localImageUrl={msg.localImageUrl} />
                </article>
              ))}
              {assistantBuffer && (
                <article className="bubble assistant">
                  <strong>{shortLabelForTextModel(modelId)}</strong>
                  <MessageBody content={assistantBuffer} />
                </article>
              )}
            </div>

            <form
              className="composer chatgpt-composer compact-composer composer-modern"
              onSubmit={sendPrompt}
              onDragEnter={onComposerDragEnter}
              onDragLeave={onComposerDragLeave}
              onDragOver={onComposerDragOver}
              onDrop={onComposerDrop}
            >
              {generatedImageUrl ? (
                <div className="composer-last-gen-strip">
                  <img className="composer-last-gen-thumb" src={generatedImageUrl} alt="" />
                  <span className="composer-last-gen-label">Last generated</span>
                </div>
              ) : null}

              <div className={`composer-card ${composerDragOver ? "composer-card--dropping" : ""}`}>
                <textarea
                  ref={textareaRef}
                  className="composer-body-input"
                  dir="auto"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={placeholder}
                  rows={2}
                  disabled={!isLoaded || isGenerating}
                />
                {imageDataUrl ? (
                  <div className="composer-attached-strip">
                    <img src={imageDataUrl} alt="" />
                    <button
                      type="button"
                      className="composer-attached-strip-remove"
                      onClick={() => setImageDataUrl(null)}
                      disabled={isGenerating}
                      aria-label="Remove image"
                    >
                      ×
                    </button>
                  </div>
                ) : null}
                <div className="composer-card-footer">
                  <div className="composer-footer-left">
                    <label
                      className="composer-mode-pill"
                      title='מוסיף הנחיה למודל לחשוב לפני התשובה. אין תצוגת "מחשבות" — רק הטקסט הסופי.'
                    >
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
                      title="מושך קטעים מוויקיפדיה (עברית/אנגלית) ומחיפוש מאגרים ב-GitHub. לא מנוע חיפוש מלא ולא גישה לכל האינטרנט."
                    >
                      <input
                        type="checkbox"
                        checked={webSearchMode}
                        onChange={(e) => setWebSearchMode(e.target.checked)}
                        disabled={isGenerating}
                      />
                      <span>Search</span>
                    </label>
                  </div>
                  <div className="composer-footer-right">
                    <button
                      type="button"
                      className="composer-clip-btn"
                      onClick={() => fileInputRef.current?.click()}
                      aria-label="Attach image"
                      title="Attach image"
                      disabled={!isLoaded || isGenerating}
                    >
                      📎
                    </button>
                    <button
                      type="submit"
                      className="composer-send-inner"
                      disabled={!isLoaded || isGenerating}
                      aria-label="Send"
                      title="Send"
                    >
                      {isGenerating ? "…" : "↑"}
                    </button>
                  </div>
                </div>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden-file-input"
                onChange={(event) => {
                  ingestImageFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </form>
            <div className="composer-footer-tools">
              <button
                type="button"
                className="text-btn"
                onClick={() => void clearModelCache()}
                disabled={isGenerating}
              >
                Clear model cache
              </button>
            </div>
          </section>
        </section>
      )}
    </main>
  );
}

export default App;
