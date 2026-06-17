import { useEffect, useRef, useState } from "react";

type Props = {
  qwenTokens: number;
  gemmaTokens: number;
  activeModel: "qwen" | "gemma" | null;
  visible: boolean;
};

function pad4(n: number): string {
  return String(Math.min(9999, Math.max(0, n))).padStart(4, "0");
}

function FloatingCount({ value, hot, tone }: { value: number; hot: boolean; tone: "qwen" | "gemma" }) {
  return (
    <span
      className={`news-token-value news-token-value--${tone}${hot ? " news-token-value--hot" : ""}`}
      aria-hidden="true"
    >
      {pad4(value)}
    </span>
  );
}

export function NewsSummaryTokenHud({ qwenTokens, gemmaTokens, activeModel, visible }: Props) {
  const [pulse, setPulse] = useState(false);
  const prevTotal = useRef(0);

  useEffect(() => {
    const total = qwenTokens + gemmaTokens;
    if (total > prevTotal.current) {
      setPulse(true);
      const t = window.setTimeout(() => setPulse(false), 140);
      prevTotal.current = total;
      return () => clearTimeout(t);
    }
    prevTotal.current = total;
  }, [qwenTokens, gemmaTokens]);

  if (!visible) return null;

  return (
    <div
      className={`news-token-hud${pulse ? " news-token-hud--pulse" : ""}`}
      role="status"
      aria-live="polite"
      aria-label={`Qwen ${qwenTokens} טוקנים, Gemma ${gemmaTokens} טוקנים`}
    >
      <div className={`news-token-lane${activeModel === "qwen" ? " news-token-lane--active" : ""}`}>
        <span className="news-token-label">QWEN</span>
        <FloatingCount value={qwenTokens} hot={activeModel === "qwen"} tone="qwen" />
      </div>
      <div className="news-token-divider" aria-hidden="true">
        <span>◆</span>
      </div>
      <div className={`news-token-lane${activeModel === "gemma" ? " news-token-lane--active" : ""}`}>
        <span className="news-token-label">GEMMA</span>
        <FloatingCount value={gemmaTokens} hot={activeModel === "gemma"} tone="gemma" />
      </div>
    </div>
  );
}
