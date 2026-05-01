import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  modelLabel?: string;
};

type WorkerOutMessage =
  | { type: "status"; text: string }
  | { type: "progress"; text: string; progress: number; detail?: string; file?: string }
  | { type: "loaded"; modelId: string; device: string }
  | { type: "caption_model_loaded"; modelId: string; device: string }
  | { type: "preload_all_done"; textModels: number; captionModels: number }
  | { type: "token"; text: string }
  | { type: "caption_done"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

const DEFAULT_MODEL = "onnx-community/gemma-4-E2B-it-ONNX";
const CODE_MODEL = "onnx-community/Qwen2.5-Coder-0.5B-Instruct-ONNX";
const MODEL_OPTIONS = [
  {
    id: "onnx-community/gemma-4-E2B-it-ONNX",
    label: "Gemma 4 E2B Instruct (Chat)",
    shortLabel: "Gemma 4",
    meta: "Google | ONNX | General",
  },
  {
    id: CODE_MODEL,
    label: "Qwen2.5 Coder 0.5B Instruct (Code)",
    shortLabel: "Qwen Coder",
    meta: "Alibaba | ONNX | Code",
  },
] as const;

const VISION_MODEL_OPTIONS = [
  { id: "Xenova/vit-gpt2-image-captioning", label: "ViT-GPT2 Captioning (Fast)" },
  { id: "onnx-community/moondream2", label: "Moondream2 (Better detail)" },
] as const;

const IMAGE_MODEL_OPTIONS = [
  { id: "flux", label: "FLUX (balanced)" },
  { id: "turbo", label: "Turbo (faster)" },
  { id: "sdxl", label: "SDXL style" },
] as const;

type ModelPreset = {
  temperature: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  topP: number;
  systemPrompt: string;
};

const DEFAULT_PRESET: ModelPreset = {
  temperature: 0.22,
  maxNewTokens: 160,
  repetitionPenalty: 1.15,
  topP: 0.92,
  systemPrompt:
    "You are a concise helpful AI assistant. Respond in the same language as the user and avoid repeating role labels. Keep answers focused and do not repeat greetings.",
};

const getModelPreset = (model: string, thinking: boolean): ModelPreset => {
  const base = { ...DEFAULT_PRESET };
  if (model.toLowerCase().includes("coder") || model.toLowerCase().includes("code")) {
    return {
      ...base,
      temperature: 0.1,
      maxNewTokens: 220,
      repetitionPenalty: 1.08,
      systemPrompt:
        "You are a senior coding assistant. Provide practical answers with correct code and brief explanations.",
    };
  }

  if (model.toLowerCase().includes("qwen")) {
    base.temperature = 0.18;
    base.maxNewTokens = 140;
    base.repetitionPenalty = 1.14;
  }

  if (thinking) {
    base.temperature = Math.max(0.15, base.temperature - 0.05);
    base.maxNewTokens += 80;
    base.systemPrompt +=
      " Think carefully step by step internally, then provide a clean final answer without exposing chain-of-thought.";
  }
  return base;
};

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
  return raw.length ? raw.slice(0, 900) : "No response generated. Try a different model.";
};

const isSimpleGreeting = (text: string): boolean => {
  const normalized = text.trim().toLowerCase();
  return /^(hi|hey|hello|shalom|שלום|היי|הי)$/.test(normalized);
};

const isCodeRequest = (text: string): boolean => {
  const normalized = text.trim().toLowerCase();
  return /(code|debug|bug|stack trace|typescript|javascript|python|function|class|compile|error)/.test(normalized);
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

function App() {
  const workerRef = useRef<Worker | null>(null);
  const assistantBufferRef = useRef("");
  const activeModelShortLabelRef = useRef("Assistant");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
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
  const [mode, setMode] = useState<"chat" | "caption" | "image">("chat");
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  const [visionReadyMap, setVisionReadyMap] = useState<Record<string, boolean>>({});
  const [preloadAllLoading, setPreloadAllLoading] = useState(false);

  const phase = isLoaded ? "ready" : isLoading ? "loading" : "start";
  const activeModelOption = useMemo(
    () => MODEL_OPTIONS.find((option) => option.id === modelId),
    [modelId],
  );
  const modelLabel = useMemo(
    () => activeModelOption?.label ?? (modelId.split("/").pop() ?? modelId),
    [activeModelOption, modelId],
  );

  useEffect(() => {
    activeModelShortLabelRef.current = activeModelOption?.shortLabel ?? "Assistant";
  }, [activeModelOption]);

  const placeholder = useMemo(() => {
    if (!isLoaded) return "Load the model first...";
    return "Ask anything...";
  }, [isLoaded]);
  const conversationTitle = useMemo(() => {
    const firstUser = messages.find((m) => m.role === "user")?.content?.trim();
    if (!firstUser) return "New chat";
    return firstUser.slice(0, 28) + (firstUser.length > 28 ? "..." : "");
  }, [messages]);

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
        setProgress(msg.progress);
        setProgressDetail(msg.detail ?? "");
        setProgressFile(msg.file ?? "");
      } else if (msg.type === "loaded") {
        setIsLoaded(true);
        setIsLoading(false);
        setProgress(100);
        setStatus(`Loaded on ${msg.device}`);
        setProgressDetail("Model ready");
        setProgressFile("");
        if (workerRef.current && !preloadAllLoading) {
          setPreloadAllLoading(true);
          setStatus("Gemma is ready. Downloading additional models in background...");
          workerRef.current.postMessage({
            type: "warmup_all",
            textModelIds: MODEL_OPTIONS.map((option) => option.id).filter((id) => id !== DEFAULT_MODEL),
            captionModelIds: VISION_MODEL_OPTIONS.map((option) => option.id),
            dtype: "q4",
          });
        }
      } else if (msg.type === "caption_model_loaded") {
        setVisionReadyMap((prev) => ({ ...prev, [msg.modelId]: true }));
        setStatus(`Vision model ready on ${msg.device}`);
      } else if (msg.type === "preload_all_done") {
        setPreloadAllLoading(false);
        const allVisionReady = Object.fromEntries(VISION_MODEL_OPTIONS.map((option) => [option.id, true]));
        setVisionReadyMap(allVisionReady);
        setIsLoading(false);
        setIsLoaded(true);
        setProgress(100);
        setProgressDetail("All local models are ready");
        setProgressFile("");
        setStatus(`All models downloaded: ${msg.textModels} text + ${msg.captionModels} vision`);
      } else if (msg.type === "token") {
        setAssistantBuffer((prev) => {
          const next = prev + msg.text;
          assistantBufferRef.current = next;
          return next;
        });
      } else if (msg.type === "caption_done") {
        setIsGenerating(false);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: msg.text,
            modelLabel: "Vision",
          },
        ]);
        setAssistantBuffer("");
        assistantBufferRef.current = "";
      } else if (msg.type === "done") {
        setIsGenerating(false);
        let output = cleanModelOutput(assistantBufferRef.current);
        if (output.length > 900) output = `${output.slice(0, 900).trimEnd()}...`;
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
    setStatus("Loading Gemma controller...");
    setProgress(0);
    setProgressDetail("Preparing local runtime...");
    setProgressFile("");
    workerRef.current.postMessage({
      type: "load",
      modelId: DEFAULT_MODEL,
      dtype: "q4",
    });
  };

  const sendPrompt = async (e: FormEvent) => {
    e.preventDefault();
    if (!workerRef.current || !isLoaded || isGenerating) return;
    const trimmed = prompt.trim();
    if (!trimmed) return;

    if (mode === "caption") {
      const captionModelId = VISION_MODEL_OPTIONS[0].id;
      if (!visionReadyMap[captionModelId]) {
        setStatus("Vision model is still loading. Please wait a few seconds.");
        return;
      }
      if (!imageDataUrl) {
        setStatus("Please attach an image first.");
        return;
      }
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: `Describe this image: ${trimmed}` }]);
      setPrompt("");
      setIsGenerating(true);
      workerRef.current.postMessage({
        type: "caption",
        imageDataUrl,
        prompt: trimmed,
        modelId: captionModelId,
      });
      return;
    }

    if (mode === "image") {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: `Create image: ${trimmed}` }]);
      setPrompt("");
      setIsGenerating(true);
      try {
        const imageModelId = IMAGE_MODEL_OPTIONS[0].id;
        const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(trimmed)}?width=1024&height=1024&model=${encodeURIComponent(imageModelId)}&nologo=true`;
        setGeneratedImageUrl(url);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `Generated image ready.\n${url}`,
            modelLabel: `Image (${imageModelId})`,
          },
        ]);
      } finally {
        setIsGenerating(false);
      }
      return;
    }

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);
    setPrompt("");
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setIsGenerating(true);

    const targetModelId = isCodeRequest(trimmed) ? CODE_MODEL : DEFAULT_MODEL;
    const preset = getModelPreset(targetModelId, thinkingMode);
    const greeting = isSimpleGreeting(trimmed);
    let webContext = "";
    if (webSearchMode) {
      setStatus("Searching web context...");
      try {
        webContext = await fetchWebContext(trimmed);
      } catch {
        webContext = "";
      }
    }
    setStatus("Generating...");

    workerRef.current.postMessage({
      type: "generate",
      modelId: targetModelId,
      prompt: trimmed,
      systemPrompt: greeting
        ? `${preset.systemPrompt} If the user sends only a greeting, reply with one short friendly sentence only.`
        : preset.systemPrompt,
      maxNewTokens: greeting ? 28 : preset.maxNewTokens,
      temperature: greeting ? 0 : preset.temperature,
      repetitionPenalty: preset.repetitionPenalty,
      topP: preset.topP,
      thinkingMode,
      webContext,
    });
  };

  return (
    <main className="app theme-space">
      <div className="bg-overlay" />
      <div className="corner-note">
        <span>Runs fully local on your browser</span>
      </div>

      {phase === "start" && (
        <section className="hero-screen glass">
          <h1>GROVEE - WEBGPU</h1>
          <p>Primary controller: Gemma 4. Image tasks run inside this interface.</p>
          <button className="pill-button" onClick={loadModel} disabled={isLoading || isGenerating}>
            {preloadAllLoading ? "Loading all models..." : "Start"}
          </button>
        </section>
      )}

      {phase === "loading" && (
        <section className="loading-screen glass">
          <h2>GROVEE - WEBGPU</h2>
          <div className="loading-model">{modelLabel}</div>
          <div className="meter">
            <div className="meter-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="percent">{progress}%</div>
          <div className="status-line">{status || "Loading model..."}</div>
          {progressDetail && <div className="status-line secondary">{progressDetail}</div>}
          {progressFile && <div className="status-line secondary">File: {progressFile}</div>}
          <div className="offline-line">Runs 100% offline</div>
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
              <div>
                <h2>{conversationTitle}</h2>
                <p className="top-status">{status}</p>
              </div>
            </header>

            <div className="messages">
              {messages.length === 0 && !assistantBuffer && (
                <div className="empty-state">
                  <h3>Message GROVEE</h3>
                  <p>Ask anything to start the conversation.</p>
                </div>
              )}
              {messages.map((msg) => (
                <article key={msg.id} className={`bubble ${msg.role}`}>
                  <strong>{msg.role === "user" ? "You" : msg.modelLabel ?? "Assistant"}</strong>
                  <p>{msg.content}</p>
                </article>
              ))}
              {assistantBuffer && (
                <article className="bubble assistant">
                  <strong>{activeModelOption?.shortLabel ?? "Assistant"}</strong>
                  <p>{assistantBuffer}</p>
                </article>
              )}
            </div>

            <div className="chat-toolbar">
              <div className="chat-controls">
                <div className="mode-group">
                  <button type="button" className={`mode-btn ${mode === "chat" ? "active" : ""}`} onClick={() => setMode("chat")}>
                    Chat
                  </button>
                  <button type="button" className={`mode-btn ${mode === "caption" ? "active" : ""}`} onClick={() => setMode("caption")}>
                    Image to Text
                  </button>
                  <button type="button" className={`mode-btn ${mode === "image" ? "active" : ""}`} onClick={() => setMode("image")}>
                    Text to Image
                  </button>
                </div>
                <span className="mode-note">Gemma 4 controller active</span>
              </div>
            </div>

            <form onSubmit={sendPrompt} className="composer chatgpt-composer">
              {mode === "caption" && (
                <div className="upload-row">
                  {imageDataUrl && <img className="preview-img" src={imageDataUrl} alt="Uploaded preview" />}
                </div>
              )}
              {mode === "image" && generatedImageUrl && (
                <div className="upload-row">
                  <img className="preview-img" src={generatedImageUrl} alt="Generated output" />
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden-file-input"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  const reader = new FileReader();
                  reader.onload = () => {
                    setImageDataUrl(String(reader.result));
                    setMode("caption");
                  };
                  reader.readAsDataURL(file);
                }}
              />

              <div className="composer-main ds-input-shell">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={
                    mode === "chat"
                      ? placeholder
                      : mode === "caption"
                        ? "What should I describe in the image?"
                        : "Describe the image you want to generate..."
                  }
                  rows={2}
                  disabled={!isLoaded || isGenerating}
                />
                <div className="ds-actions-row">
                  <div className="ds-left-actions">
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={thinkingMode}
                        onChange={(e) => setThinkingMode(e.target.checked)}
                        disabled={isGenerating}
                      />
                      <span>DeepThink</span>
                    </label>
                    <label className="toggle">
                      <input
                        type="checkbox"
                        checked={webSearchMode}
                        onChange={(e) => setWebSearchMode(e.target.checked)}
                        disabled={isGenerating}
                      />
                      <span>Search</span>
                    </label>
                  </div>
                  <div className="ds-right-actions">
                    <button
                      type="button"
                      className="icon-btn subtle-btn"
                      onClick={() => fileInputRef.current?.click()}
                      aria-label="Attach image"
                      title="Attach image"
                    >
                      📎
                    </button>
                    <button className="send-btn" type="submit" disabled={!isLoaded || isGenerating}>
                      {isGenerating ? "..." : "↑"}
                    </button>
                  </div>
                </div>
              </div>
            </form>
          </section>
        </section>
      )}
    </main>
  );
}

export default App;
