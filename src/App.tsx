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
  | { type: "progress"; text: string; progress: number }
  | { type: "loaded"; modelId: string; device: string }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

const DEFAULT_MODEL = "onnx-community/gemma-4-E2B-it-ONNX";
const MODEL_OPTIONS = [
  {
    id: "onnx-community/gemma-4-E2B-it-ONNX",
    label: "Gemma 4 E2B Instruct",
    shortLabel: "Gemma 4",
    meta: "Google | ONNX | General",
  },
  {
    id: "onnx-community/Qwen3-0.6B-ONNX",
    label: "Qwen3 0.6B",
    shortLabel: "Qwen3",
    meta: "Alibaba | ONNX | Fast",
  },
  {
    id: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    label: "Llama 3.2 1B Instruct",
    shortLabel: "Llama 3.2",
    meta: "Meta | ONNX | General",
  },
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

  return normalized.length ? normalized : "I could not generate a stable response. Please try again.";
};

const isSimpleGreeting = (text: string): boolean => {
  const normalized = text.trim().toLowerCase();
  return /^(hi|hey|hello|shalom|שלום|היי|הי)$/.test(normalized);
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
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState("Not loaded");
  const [progress, setProgress] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [assistantBuffer, setAssistantBuffer] = useState("");
  const [thinkingMode, setThinkingMode] = useState(false);
  const [webSearchMode, setWebSearchMode] = useState(false);

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
      } else if (msg.type === "loaded") {
        setIsLoaded(true);
        setIsLoading(false);
        setProgress(100);
        setStatus(`Loaded on ${msg.device}`);
      } else if (msg.type === "token") {
        setAssistantBuffer((prev) => {
          const next = prev + msg.text;
          assistantBufferRef.current = next;
          return next;
        });
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
        setProgress(0);
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
    setIsLoading(true);
    setStatus("Loading model...");
    setProgress(0);
    workerRef.current.postMessage({
      type: "load",
      modelId,
      dtype: "q4",
    });
  };

  const sendPrompt = async (e: FormEvent) => {
    e.preventDefault();
    if (!workerRef.current || !isLoaded || isGenerating) return;
    const trimmed = prompt.trim();
    if (!trimmed) return;

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);
    setPrompt("");
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setIsGenerating(true);

    const preset = getModelPreset(modelId, thinkingMode);
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
          <p>AI models running directly on your device with WebGPU</p>
          <div className="selector-row">
            <select
              id="model"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={isLoading || isGenerating}
            >
              {MODEL_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label} - {option.meta}
                </option>
              ))}
            </select>
          </div>
          <button className="pill-button" onClick={loadModel} disabled={isLoading || isGenerating}>
            Load model
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
          <div className="offline-line">Runs 100% offline</div>
        </section>
      )}

      {phase === "ready" && (
        <section className="chat-screen">
          <header className="chat-top glass">
            <div className="title-wrap">
              <h2>GROVEE - WEBGPU</h2>
              <p className="top-status">{status}</p>
            </div>
            <div className="top-actions" />
          </header>

          <section className="chat glass">
            <div className="messages">
              {messages.length === 0 && !assistantBuffer && (
                <div className="empty-state">
                  <h3>Start a conversation</h3>
                  <p>Ask a question, summarize text, or request code help.</p>
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

            <form onSubmit={sendPrompt} className="composer">
              <div className="chat-controls">
                <select
                  id="model"
                  className="model-select"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  disabled={isLoading || isGenerating}
                >
                  {MODEL_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label} - {option.meta}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="reload-btn"
                  onClick={loadModel}
                  disabled={isLoading || isGenerating}
                >
                  Reload
                </button>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={thinkingMode}
                    onChange={(e) => setThinkingMode(e.target.checked)}
                    disabled={isGenerating}
                  />
                  <span>Thinking</span>
                </label>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={webSearchMode}
                    onChange={(e) => setWebSearchMode(e.target.checked)}
                    disabled={isGenerating}
                  />
                  <span>Web Search</span>
                </label>
              </div>
              <div className="composer-main">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={placeholder}
                  rows={4}
                  disabled={!isLoaded || isGenerating}
                />
                <div className="composer-hint">Shift+Enter for new line</div>
              </div>
              <button type="submit" disabled={!isLoaded || isGenerating}>
                {isGenerating ? "Generating..." : "Send"}
              </button>
            </form>
          </section>
        </section>
      )}
    </main>
  );
}

export default App;
