import { useEffect, useRef, useState } from "react";

type Props = {
  active: boolean;
  /** he → Hebrew labels; anything else → English. */
  rtl?: boolean;
  /** Compact variant for quad tiles (smaller type, hidden sub-labels). */
  compact?: boolean;
};

/**
 * Antenna auto-tuning meter shown over the analog snow while a channel locks on.
 * Mimics an old TV set scanning the band: a sweeping signal bar, a climbing
 * signal-strength percentage, and a flickering channel/frequency readout.
 */
export function TvTuningScan({ active, rtl = false, compact = false }: Props) {
  const [pct, setPct] = useState(0);
  const [freq, setFreq] = useState(48.25);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);

  useEffect(() => {
    if (!active) {
      setPct(0);
      return;
    }
    startRef.current = performance.now();
    setPct(0);

    const tick = (t: number) => {
      const elapsed = t - startRef.current;
      // Ease toward ~96% over ~1.4s; final lock-on jump handled by unmount.
      const target = Math.min(96, (elapsed / 1400) * 100);
      // Jittered climb so the meter feels like a real signal search.
      setPct((prev) => {
        const next = prev + (target - prev) * 0.25 + (Math.random() * 3 - 1.5);
        return Math.max(0, Math.min(99, next));
      });
      setFreq(48.25 + Math.random() * 807.75);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [active]);

  if (!active) return null;

  const bars = compact ? 10 : 18;
  const litBars = Math.round((pct / 100) * bars);
  const label = rtl ? "מכוון ערוץ" : "TUNING CHANNEL";
  const sub = rtl ? "סורק את התדרים…" : "Scanning frequencies…";

  return (
    <div
      className={`lm-tune-scan${compact ? " lm-tune-scan--compact" : ""}`}
      aria-hidden="true"
    >
      <div className="lm-tune-scan__sweep" />
      <div className="lm-tune-scan__panel">
        <div className="lm-tune-scan__row">
          <span className="lm-tune-scan__label">{label}</span>
          <span className="lm-tune-scan__freq">{freq.toFixed(2)} MHz</span>
        </div>
        <div className="lm-tune-scan__meter">
          {Array.from({ length: bars }, (_, i) => (
            <span
              key={i}
              className={`lm-tune-scan__bar${i < litBars ? " is-lit" : ""}`}
              style={{ height: `${30 + (i / bars) * 70}%` }}
            />
          ))}
        </div>
        <div className="lm-tune-scan__row lm-tune-scan__row--foot">
          <span className="lm-tune-scan__sub">{sub}</span>
          <span className="lm-tune-scan__pct">{Math.round(pct)}%</span>
        </div>
      </div>
    </div>
  );
}
