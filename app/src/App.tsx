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
};

type WorkerOutMessage =
  | { type: "status"; text: string }
  | { type: "progress"; text: string; progress: number; detail?: string; file?: string }
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
/** Public ONNX repo for Transformers.js (the *-Instruct-ONNX* repo returns 401 for anonymous fetch). */
const CODE_MODEL = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";
const MODEL_CACHE_FLAG = "grovee_models_warmed_v1";
const SETTINGS_STORAGE_KEY = "grovee_model_settings_v1";

const VISION_MODEL_OPTIONS = [
  { id: "Xenova/vit-gpt2-image-captioning", label: "ViT-GPT2 Captioning (Fast)" },
  { id: "onnx-community/moondream2", label: "Moondream2 (Better detail)" },
] as const;

/** Chat + warmup only use the fast caption model; Moondream is omitted to avoid multi‑GB RAM / OOM on refresh. */
const DEFAULT_CAPTION_MODEL_ID = VISION_MODEL_OPTIONS[0].id;

const IMAGE_MODEL_OPTIONS = [
  { id: "flux", label: "FLUX (balanced)" },
  { id: "turbo", label: "Turbo (faster)" },
  { id: "sdxl", label: "SDXL style" },
] as const;

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

type ImageProviderId = "pollinations" | "local_sd_turbo";

type AppSettings = {
  gemma: TunableModelSettings;
  coder: TunableModelSettings;
  visionMaxTokens: number;
  imageBackendModel: string;
  imageProvider: ImageProviderId;
};

const defaultAppSettings = (): AppSettings => ({
  gemma: { ...defaultGemmaSettings },
  coder: { ...defaultCoderSettings },
  visionMaxTokens: 96,
  imageBackendModel: IMAGE_MODEL_OPTIONS[0].id,
  imageProvider: "pollinations",
});

const loadSettings = (): AppSettings => {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return defaultAppSettings();
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      ...defaultAppSettings(),
      ...parsed,
      gemma: { ...defaultGemmaSettings, ...parsed.gemma },
      coder: { ...defaultCoderSettings, ...parsed.coder },
      imageProvider:
        parsed.imageProvider === "local_sd_turbo" || parsed.imageProvider === "pollinations"
          ? parsed.imageProvider
          : defaultAppSettings().imageProvider,
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

type ImageOrch =
  | { step: "translate"; userText: string }
  | { step: "summarize"; userText: string; englishPrompt: string; imageUrl: string };

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

const fetchWebContext = async (query: string): Promise<string> => {
  const encoded = encodeURIComponent(query);
  const endpoint = `https://en.wikipedia.org/w/api.php?action=opensearch&search=${encoded}&limit=3&namespace=0&format=json&origin=*`;
  const response = await fetch(endpoint);
  if (!response.ok) throw new Error("Web lookup failed");
  const data = (await response.json()) as [string, string[], string[], string[]];
  const titles = data[1] ?? [];
  const snippets = data[2] ?? [];
  const urls = data[3] ?? [];
  if (!titles.length) return "";
  return titles
    .map((title, i) => `- ${title}: ${snippets[i] ?? ""} (${urls[i] ?? ""})`)
    .join("\n");
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

function MessageBody({ content }: { content: string }) {
  const parts = useMemo(() => extractRichParts(content), [content]);
  const dir = isRtlText(content) ? "rtl" : "ltr";

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
          Gemma 4 מנהל את השיחה והעברית; לקוד הוא מפיק משימה טכנית באנגלית ואז <strong>Qwen Coder</strong> כותב את הקוד (Gemma
          יכולה לנחש קוד קצר, אבל לא זה התפקיד שלה). תמונות: פרומפט באנגלית אוטומטית, מנוע ענן או SD-Turbo מקומי.
        </p>

        <section className="settings-section">
          <h3>Gemma 4 (controller / chat)</h3>
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
          <label className="settings-row block">
            <span>Engine</span>
            <select
              value={draft.imageProvider}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  imageProvider: e.target.value as ImageProviderId,
                }))
              }
            >
              <option value="pollinations">Cloud — Pollinations (fast, needs network)</option>
              <option value="local_sd_turbo">Local — SD-Turbo in browser ({getSdTurboSizeNote()})</option>
            </select>
          </label>
          <p className="settings-micro">
            Smaller experimental ONNX demos (under ~400MB) exist in the community, but this app ships with SD-Turbo as the
            practical single-step local option. See{" "}
            <a href="https://github.com/lacerbi/web-txt2img" target="_blank" rel="noreferrer">
              web-txt2img
            </a>{" "}
            and{" "}
            <a
              href="https://huggingface.co/IlyasMoutawwakil/tiny-stable-diffusion-onnx"
              target="_blank"
              rel="noreferrer"
            >
              tiny-stable-diffusion-onnx
            </a>{" "}
            for research builds.
          </p>
          {draft.imageProvider === "pollinations" && (
            <label className="settings-row block">
              <span>Pollinations model id</span>
              <select
                value={draft.imageBackendModel}
                onChange={(e) => setDraft((d) => ({ ...d, imageBackendModel: e.target.value }))}
              >
                {IMAGE_MODEL_OPTIONS.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          )}
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

  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [appSettings, setAppSettings] = useState<AppSettings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsModalKey, setSettingsModalKey] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState("Not loaded");
  const [progress, setProgress] = useState(0);
  const [progressDetail, setProgressDetail] = useState("");
  const [progressFile, setProgressFile] = useState("");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [assistantBuffer, setAssistantBuffer] = useState("");
  const [thinkingMode, setThinkingMode] = useState(false);
  const [webSearchMode, setWebSearchMode] = useState(false);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  const lastGeneratedUrlRef = useRef<string | null>(null);
  const [composerDragOver, setComposerDragOver] = useState(false);
  const [preloadAllLoading, setPreloadAllLoading] = useState(false);
  const [shouldWarmupOnStart, setShouldWarmupOnStart] = useState(() => {
    try {
      return localStorage.getItem(MODEL_CACHE_FLAG) !== "1";
    } catch {
      return true;
    }
  });
  const appSettingsRef = useRef(appSettings);
  const thinkingRef = useRef(thinkingMode);
  const webSearchRef = useRef(webSearchMode);
  const preloadAllLoadingRef = useRef(preloadAllLoading);
  const shouldWarmupOnStartRef = useRef(shouldWarmupOnStart);
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
    preloadAllLoadingRef.current = preloadAllLoading;
  }, [preloadAllLoading]);
  useEffect(() => {
    shouldWarmupOnStartRef.current = shouldWarmupOnStart;
  }, [shouldWarmupOnStart]);
  useEffect(() => {
    isLoadingRef.current = isLoading;
  }, [isLoading]);

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
    activeModelShortLabelRef.current = "Gemma 4";
  }, []);

  const placeholder = useMemo(() => {
    if (!isLoaded) return "Start the app first…";
    return "Ask anything…";
  }, [isLoaded]);
  const composerHint =
    "Hebrew or English · + or drag an image · code or image requests — עברית או אנגלית · צרף תמונה בגרירה";
  const conversationTitle = useMemo(() => {
    const firstUser = messages.find((m) => m.role === "user")?.content?.trim();
    if (!firstUser) return "New chat";
    return firstUser.slice(0, 28) + (firstUser.length > 28 ? "…" : "");
  }, [messages]);

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
      modelId: DEFAULT_MODEL,
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
    const worker = new Worker(new URL("./model.worker.ts", import.meta.url), {
      type: "module",
    });

    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;
      if (msg.type === "status") {
        setStatus(msg.text);
      } else if (msg.type === "progress") {
        setStatus(msg.text);
        const loadingPhase = isLoadingRef.current || preloadAllLoadingRef.current;
        setProgress((prev) => (loadingPhase ? Math.max(prev, msg.progress) : msg.progress));
        setProgressDetail(msg.detail ?? "");
        setProgressFile(msg.file ?? "");
      } else if (msg.type === "loaded") {
        setIsLoaded(true);
        setIsLoading(false);
        setProgress(100);
        setStatus(`Loaded on ${msg.device}`);
        setProgressDetail("Model ready");
        setProgressFile("");
        if (workerRef.current && !preloadAllLoadingRef.current && shouldWarmupOnStartRef.current) {
          setPreloadAllLoading(true);
          setStatus("Gemma is ready. Warming vision model (weights stay in browser cache)…");
          workerRef.current.postMessage({
            type: "warmup_all",
            textModelIds: [],
            captionModelIds: [DEFAULT_CAPTION_MODEL_ID],
            dtype: "q4",
          });
        } else {
          setStatus(`Gemma controller ready on ${msg.device}`);
          setProgressDetail("Using Hugging Face cache in this browser");
          setProgressFile("");
          workerRef.current?.postMessage({ type: "preload_caption", modelId: DEFAULT_CAPTION_MODEL_ID });
        }
      } else if (msg.type === "caption_model_loaded") {
        setStatus(`Vision model ready on ${msg.device}`);
      } else if (msg.type === "preload_all_done") {
        setPreloadAllLoading(false);
        setIsLoading(false);
        setIsLoaded(true);
        setShouldWarmupOnStart(false);
        try {
          localStorage.setItem(MODEL_CACHE_FLAG, "1");
        } catch {
          // ignore
        }
        setProgress(100);
        const failedT = msg.failedTextModelIds?.length
          ? ` · נכשלו מודלי טקסט: ${msg.failedTextModelIds.join(", ")}`
          : "";
        const failedC = msg.failedCaptionModelIds?.length
          ? ` · נכשל vision: ${msg.failedCaptionModelIds.join(", ")}`
          : "";
        setProgressDetail(
          failedT || failedC
            ? "חלק מהמודלים לא נטענו — בדקו רשת/חסימות Hugging Face"
            : "מודל הכיתוב מוכן; Qwen ייטען בפעם הראשונה שתבקשו קוד",
        );
        setProgressFile("");
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
              modelLabel: "Gemma 4",
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

          const cloudUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(english)}?width=1024&height=1024&model=${encodeURIComponent(appSettingsRef.current.imageBackendModel)}&nologo=true`;

          const runSummarize = (imageUrl: string) => {
            setGeneratedImageUrl(imageUrl);
            orchRef.current = {
              kind: "image",
              data: {
                step: "summarize",
                userText,
                englishPrompt: english,
                imageUrl,
              },
            };
            setStatus("Gemma: replying in your language…");
            const g = appSettingsRef.current.gemma;
            const userLang = isRtlText(userText) ? "Hebrew" : "the same language as the user";
            runGemmaGenerate(
              `User request:\n${userText}\n\nWe generated an image using this English prompt for the image model:\n${english}\n\nReply in ${userLang}: 2–4 sentences explaining what was created; mention that the prompt was translated for the image engine. Then add a line with the image URL:\n${imageUrl}`,
              g.systemPrompt,
              Math.min(320, g.maxNewTokens),
              g.temperature,
              g.repetitionPenalty,
              g.topP,
              "",
            );
          };

          if (appSettingsRef.current.imageProvider === "local_sd_turbo") {
            void (async () => {
              const local = await generateSdTurboPng(english, (s) => setStatus(s));
              if (local.ok) {
                runSummarize(local.objectUrl);
              } else {
                setStatus(`Local image failed (${local.message}). Using cloud…`);
                runSummarize(cloudUrl);
              }
            })();
          } else {
            runSummarize(cloudUrl);
          }
          return;
        }

        if (orch?.kind === "image" && orch.data.step === "summarize") {
          const summary = buf || "Image ready.";
          const imageUrl = orch.data.imageUrl;
          orchRef.current = null;
          setIsGenerating(false);
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: `${summary}\n\n![Generated](${imageUrl})`,
              modelLabel: "Gemma 4",
            },
          ]);
          setAssistantBuffer("");
          assistantBufferRef.current = "";
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
              modelLabel: "Gemma 4",
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
        setPreloadAllLoading(false);
        setProgress(0);
        setProgressDetail("");
        setProgressFile("");
        setStatus(`Error: ${msg.error}`);
      }
    };

    workerRef.current = worker;
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const loadModel = () => {
    if (!workerRef.current) return;
    setModelId(DEFAULT_MODEL);
    setIsLoading(true);
    setPreloadAllLoading(false);
    setStatus("Loading Gemma controller…");
    setProgress(0);
    setProgressDetail("Preparing local runtime…");
    setProgressFile("");
    workerRef.current.postMessage({
      type: "load",
      modelId: DEFAULT_MODEL,
      dtype: "q4",
    });
  };

  const clearModelCache = async () => {
    if (isGenerating) return;
    setStatus("Clearing local model cache…");
    try {
      terminateLocalImageWorker();
      revokeImageUrl(generatedImageUrl);
      setGeneratedImageUrl(null);
      lastGeneratedUrlRef.current = null;
      workerRef.current?.postMessage({ type: "clear_runtime_cache" });
      localStorage.removeItem(MODEL_CACHE_FLAG);
      setShouldWarmupOnStart(true);
      orchRef.current = null;

      if ("caches" in window) {
        const cacheKeys = await caches.keys();
        await Promise.all(
          cacheKeys
            .filter((key) => key.toLowerCase().includes("transformers") || key.toLowerCase().includes("huggingface"))
            .map((key) => caches.delete(key)),
        );
      }

      const indexedDbGlobal = indexedDB as IDBFactory & {
        databases?: () => Promise<Array<{ name?: string }>>;
      };
      if (indexedDbGlobal.databases) {
        const dbs = await indexedDbGlobal.databases();
        for (const db of dbs) {
          const name = db.name ?? "";
          if (!name) continue;
          const lower = name.toLowerCase();
          if (lower.includes("transformers") || lower.includes("huggingface") || lower.includes("onnx")) {
            indexedDB.deleteDatabase(name);
          }
        }
      }
      setStatus("Cache cleared. Press Start again.");
      setProgress(0);
      setProgressDetail("");
      setProgressFile("");
      setIsLoaded(false);
      setIsLoading(false);
      setPreloadAllLoading(false);
      setMessages([]);
      setAssistantBuffer("");
      assistantBufferRef.current = "";
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`Cache clear failed: ${message}`);
    }
  };

  const persistSettings = (s: AppSettings) => {
    setAppSettings(s);
    saveSettings(s);
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
    if (webSearchMode) {
      setStatus("Searching…");
      try {
        webContext = await fetchWebContext(trimmed);
      } catch {
        webContext = "";
      }
    }
    setStatus("Generating…");

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
              {preloadAllLoading ? "מסיים…" : "התחל"}
            </button>
            <button className="pill-button subtle-btn" onClick={clearModelCache} disabled={isGenerating || isLoading}>
              נקה מטמון
            </button>
          </div>
        </section>
      )}

      {phase === "loading" && (
        <section className="loading-screen glass">
          <h2>GROVEE</h2>
          <div className="loading-model">{modelLabel}</div>
          <div className="meter">
            <div
              className={`meter-fill${progress >= 94 && progress < 100 ? " meter-fill-await" : ""}`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="percent">{progress}%</div>
          <div className="status-line">{status || "Loading…"}</div>
          {progress >= 90 && progress < 100 && (
            <div className="loading-final-phase" role="status">
              <p className="loading-final-phase-title">לא תקוע — שלב אחרון</p>
              <p className="loading-final-phase-body">
                כשההורדה מהרשת נגמרת, האחוז עלול להישאר סביב <strong>99%</strong> בזמן שהדפדפן מרכיב את גרף ONNX
                (WebGPU או WASM) וטוען את המודל לזיכרון. זה יכול לקחת מ־כמה שניות ועד כ־שתי דקות במחשבים איטיים.
                <span className="loading-final-phase-nowrap"> אין צורך לרענן את הדף.</span>
              </p>
            </div>
          )}
          {progressDetail && <div className="status-line secondary">{progressDetail}</div>}
          {progressFile && <div className="status-line secondary">File: {progressFile}</div>}
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
                setMessages([]);
                setAssistantBuffer("");
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
              <button type="button" className="chat-item active">
                {conversationTitle}
              </button>
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
                  <MessageBody content={msg.content} />
                </article>
              ))}
              {assistantBuffer && (
                <article className="bubble assistant">
                  <strong>Gemma 4</strong>
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

              <div className="composer-shell">
                <div className={`composer-card ${composerDragOver ? "composer-card--dropping" : ""}`}>
                  <div className="composer-input-line">
                    <button
                      type="button"
                      className="composer-lead-btn"
                      onClick={() => fileInputRef.current?.click()}
                      aria-label="Attach image"
                      title="Attach image"
                      disabled={!isLoaded || isGenerating}
                    >
                      +
                    </button>
                    {imageDataUrl ? (
                      <div className="composer-inline-thumb">
                        <img src={imageDataUrl} alt="" />
                        <button
                          type="button"
                          className="composer-inline-thumb-remove"
                          onClick={() => setImageDataUrl(null)}
                          disabled={isGenerating}
                          aria-label="Remove image"
                        >
                          ×
                        </button>
                      </div>
                    ) : null}
                    <textarea
                      ref={textareaRef}
                      className="composer-field"
                      dir="auto"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder={placeholder}
                      title={composerHint}
                      rows={1}
                      disabled={!isLoaded || isGenerating}
                    />
                  </div>
                  <div className="composer-card-toolbar" aria-label="Options">
                    <label className="composer-mode-pill">
                      <input
                        type="checkbox"
                        checked={thinkingMode}
                        onChange={(e) => setThinkingMode(e.target.checked)}
                        disabled={isGenerating}
                      />
                      <span>Think</span>
                    </label>
                    <label className="composer-mode-pill">
                      <input
                        type="checkbox"
                        checked={webSearchMode}
                        onChange={(e) => setWebSearchMode(e.target.checked)}
                        disabled={isGenerating}
                      />
                      <span>Search</span>
                    </label>
                  </div>
                </div>
                <button
                  type="submit"
                  className="composer-send-fab"
                  disabled={!isLoaded || isGenerating}
                  aria-label="Send"
                  title="Send"
                >
                  {isGenerating ? "…" : "↑"}
                </button>
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
                onClick={clearModelCache}
                disabled={isGenerating || isLoading}
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
