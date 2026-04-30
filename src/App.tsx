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

  const placeholder = useMemo(() => {
    if (!isLoaded) return "Load the model first...";
    return "Write your prompt here...";
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
    <main className="app">
      <header className="topbar">
        <div>
          <h1>GROVEE Model WebGPU</h1>
          <p>Run Gemma-style ONNX models directly in browser using WebGPU.</p>
        </div>
      </header>

      <section className="panel">
        <label className="label" htmlFor="model">
          Model
        </label>
        <div className="row">
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
            {isLoading ? "Loading..." : "Load model"}
          </button>
        </div>
        <p className="status">{status}</p>
        <progress max={100} value={progress} />
      </section>

      <section className="chat">
        <div className="messages">
          {messages.map((msg) => (
            <article key={msg.id} className={`bubble ${msg.role}`}>
              <strong>{msg.role === "user" ? "You" : "Assistant"}</strong>
              <p>{msg.content}</p>
            </article>
          ))}
          {assistantBuffer && (
            <article className="bubble assistant">
              <strong>Assistant</strong>
              <p>{assistantBuffer}</p>
            </article>
          )}
        </div>

        <form onSubmit={sendPrompt} className="composer">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={placeholder}
            rows={4}
            disabled={!isLoaded || isGenerating}
          />
          <button type="submit" disabled={!isLoaded || isGenerating}>
            {isGenerating ? "Generating..." : "Send"}
          </button>
        </form>
      </section>
    </main>
  );
}

export default App;
