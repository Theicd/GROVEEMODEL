import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
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
  "onnx-community/gemma-4-E2B-it-ONNX",
  "onnx-community/Qwen3-0.6B-ONNX",
];

function App() {
  const workerRef = useRef<Worker | null>(null);
  const assistantBufferRef = useRef("");
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState("Not loaded");
  const [progress, setProgress] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [assistantBuffer, setAssistantBuffer] = useState("");

  const phase = isLoaded ? "ready" : isLoading ? "loading" : "start";

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
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "assistant", content: assistantBufferRef.current.trim() },
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

  const sendPrompt = (e: FormEvent) => {
    e.preventDefault();
    if (!workerRef.current || !isLoaded || isGenerating) return;
    const trimmed = prompt.trim();
    if (!trimmed) return;

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);
    setPrompt("");
    setAssistantBuffer("");
    assistantBufferRef.current = "";
    setIsGenerating(true);

    workerRef.current.postMessage({
      type: "generate",
      prompt: trimmed,
      systemPrompt:
        "You are a concise helpful assistant. Answer clearly and practically in the same language as the user.",
      maxNewTokens: 256,
      temperature: 0.3,
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
          <h1>Gemma 4 WebGPU</h1>
          <p>Multimodal AI, running locally in your browser with WebGPU</p>
          <div className="selector-row">
            <select
              id="model"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={isLoading || isGenerating}
            >
              {MODEL_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
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
          <h2>Gemma 4 WebGPU</h2>
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
              <h2>Gemma 4 WebGPU</h2>
              <p className="top-status">{status}</p>
            </div>
            <div className="top-actions">
              <span className="model-pill">{modelId}</span>
              <select
                id="model"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                disabled={isLoading || isGenerating}
              >
                {MODEL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <button onClick={loadModel} disabled={isLoading || isGenerating}>
                Reload model
              </button>
            </div>
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
                  <strong>{msg.role === "user" ? "You" : "Gemma"}</strong>
                  <p>{msg.content}</p>
                </article>
              ))}
              {assistantBuffer && (
                <article className="bubble assistant">
                  <strong>Gemma</strong>
                  <p>{assistantBuffer}</p>
                </article>
              )}
            </div>

            <form onSubmit={sendPrompt} className="composer">
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
