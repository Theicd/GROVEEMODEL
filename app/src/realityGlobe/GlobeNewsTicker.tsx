import { useEffect, useMemo, useRef } from "react";
import type { IntelTickerItem } from "./intelFeed";
import { formatRelativeTimeHe, iconForCategory } from "./tickerUtils";

type Props = {
  items: IntelTickerItem[];
  headlines?: { id: string; text: string; severity: number }[];
  loading?: boolean;
  onItemClick?: (item: IntelTickerItem) => void;
};

const sevColor = (s: number): string => {
  if (s >= 5) return "#ff1744";
  if (s >= 4) return "#ff9100";
  if (s >= 3) return "#ffd600";
  return "#00e5ff";
};

const FALLBACK: IntelTickerItem[] = [
  {
    id: "waiting",
    severity: 1,
    tag: "LIVE",
    text: "ממתין לנתונים מהגלובוס…",
    time: "",
    ts: Date.now(),
    icon: "📡",
  },
];

export function GlobeNewsTicker({ items, headlines = [], loading, onItemClick }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const offsetRef = useRef(0);

  const line = useMemo(() => {
    const headlineItems: IntelTickerItem[] = headlines.map((h) => ({
      id: `hl-${h.id}`,
      severity: h.severity,
      tag: h.severity >= 5 ? "BREAKING" : "כותרת",
      text: h.text,
      time: "",
      ts: Date.now(),
      icon: h.severity >= 5 ? "🔴" : "📰",
      category: "HEADLINE",
    }));
    const merged = [...headlineItems, ...items];
    const src = loading && !merged.length ? FALLBACK : merged.length ? merged : FALLBACK;
    return src.map((item) => ({
      ...item,
      ts: item.ts ?? Date.now(),
      icon: item.icon || iconForCategory(item.tag, item.category),
    }));
  }, [items, headlines, loading]);

  const track = useMemo(() => [...line, ...line], [line]);

  useEffect(() => {
    offsetRef.current = 0;
  }, [line.length]);

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    let raf = 0;
    const speed = 0.6;

    const tick = () => {
      const half = el.scrollWidth / 2;
      if (half > 0) {
        offsetRef.current -= speed;
        if (offsetRef.current <= -half) offsetRef.current = 0;
        el.style.transform = `translate3d(${offsetRef.current}px,0,0)`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [track.length]);

  const handleClick = (item: IntelTickerItem) => {
    onItemClick?.(item);
  };

  return (
    <div className="globe-news-ticker" aria-live="polite">
      <div className="globe-news-ticker-label">
        <span className="globe-news-ticker-live-dot" aria-hidden="true" />
        LIVE
      </div>
      <div className="globe-news-ticker-viewport">
        <div ref={trackRef} className="globe-news-ticker-track">
          {track.map((item, i) => {
            const rel = formatRelativeTimeHe(item.ts ?? Date.now());
            const clickable = Boolean(
              onItemClick &&
                (item.lat != null ||
                  item.lon != null ||
                  item.geo ||
                  item.category === "ISRAEL" ||
                  item.category === "HEADLINE"),
            );
            return (
              <span
                key={`${item.id}-${i}`}
                className={`globe-news-ticker-item${clickable ? " globe-news-ticker-item--clickable" : ""}`}
                role={clickable ? "button" : undefined}
                tabIndex={clickable ? 0 : undefined}
                onClick={clickable ? () => handleClick(item) : undefined}
                onKeyDown={
                  clickable
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleClick(item);
                        }
                      }
                    : undefined
                }
              >
                <span className="globe-news-ticker-icon" aria-hidden="true">
                  {item.icon}
                </span>
                <span
                  className="globe-news-ticker-tag"
                  style={{ background: sevColor(item.severity), color: "#030810" }}
                >
                  {item.tag}
                </span>
                <span className="globe-news-ticker-text">{item.text}</span>
                <span className="globe-news-ticker-time" title={item.time}>
                  {rel}
                  {item.time ? ` · ${item.time}` : ""}
                </span>
                <span className="globe-news-ticker-sep" aria-hidden="true">
                  ◆
                </span>
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
