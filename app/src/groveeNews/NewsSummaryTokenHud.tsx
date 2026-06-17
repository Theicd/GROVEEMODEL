import { useEffect, useRef, useState } from "react";

type Props = {
  gemmaTokens: number;
  active: boolean;
  visible: boolean;
};

function pad4(n: number): string {
  return String(Math.min(9999, Math.max(0, n))).padStart(4, "0");
}

export function NewsSummaryTokenHud({ gemmaTokens, active, visible }: Props) {
  const [pulse, setPulse] = useState(false);
  const prev = useRef(0);

  useEffect(() => {
    if (gemmaTokens > prev.current) {
      setPulse(true);
      const t = window.setTimeout(() => setPulse(false), 140);
      prev.current = gemmaTokens;
      return () => clearTimeout(t);
    }
    prev.current = gemmaTokens;
  }, [gemmaTokens]);

  if (!visible) return null;

  return (
    <div
      className={`news-token-hud news-token-hud--gemma-only${pulse ? " news-token-hud--pulse" : ""}`}
      role="status"
      aria-live="polite"
      aria-label={`Gemma ${gemmaTokens} טוקנים`}
    >
      <div className={`news-token-lane${active ? " news-token-lane--active" : ""}`}>
        <span className="news-token-label">GEMMA</span>
        <span
          className={`news-token-value news-token-value--gemma${active ? " news-token-value--hot" : ""}`}
          aria-hidden="true"
        >
          {pad4(gemmaTokens)}
        </span>
      </div>
    </div>
  );
}
