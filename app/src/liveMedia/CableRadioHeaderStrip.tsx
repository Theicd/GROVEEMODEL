import { useCallback, useEffect, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "../searchResults/types";
import { HlsStreamPlayer } from "../searchResults/HlsStreamPlayer";

type Props = {
  stations: UnifiedSearchHit[];
  activeIndex: number;
  animPhase: "in" | "out";
  uiLang: ChatUiLanguage;
  audioFocus: boolean;
  muted: boolean;
  volume: number;
  onActivate: () => void;
  onOpenFull: (index: number) => void;
  onSelectIndex: (index: number) => void;
};

export function CableRadioHeaderStrip({
  stations,
  activeIndex,
  animPhase,
  uiLang,
  audioFocus,
  muted,
  volume,
  onActivate,
  onOpenFull,
  onSelectIndex,
}: Props) {
  const rtl = uiLang === "he";
  const total = stations.length;
  const hit = stations[activeIndex] ?? null;
  const prevIdx = total > 1 ? (activeIndex - 1 + total) % total : -1;
  const nextIdx = total > 1 ? (activeIndex + 1) % total : -1;
  const prevHit = prevIdx >= 0 ? stations[prevIdx] : null;
  const nextHit = nextIdx >= 0 ? stations[nextIdx] : null;
  const [playing, setPlaying] = useState(false);
  const src = hit?.mediaPlayUrl || hit?.url || "";
  const onReady = useCallback(() => setPlaying(true), []);

  useEffect(() => {
    setPlaying(false);
  }, [hit?.id, src]);

  const L =
    uiLang === "he"
      ? { open: "פתח רדיו", live: "שידור חי" }
      : { open: "Open radio", live: "Live" };

  if (!hit || total < 1) return null;

  return (
    <div className="lm-cable-radio-header" dir={rtl ? "rtl" : "ltr"}>
      <div className="lm-cable-radio-header-stage" aria-live="polite">
        {prevHit ? (
          <button
            type="button"
            className="lm-cable-radio-header-peek lm-cable-radio-header-peek--prev"
            onClick={() => onSelectIndex(prevIdx)}
            aria-label={prevHit.title}
          >
            {prevHit.imageUrl ? (
              <img src={prevHit.imageUrl} alt="" referrerPolicy="no-referrer" />
            ) : (
              <span>📻</span>
            )}
          </button>
        ) : null}

        <div
          className={`lm-cable-radio-header-slide lm-cable-radio-header-slide--${animPhase}${playing ? " is-playing" : ""}${audioFocus ? " is-audio-focus" : ""}`}
          onClick={onActivate}
          onDoubleClick={(e) => {
            e.preventDefault();
            onOpenFull(activeIndex);
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter") onOpenFull(activeIndex);
          }}
        >
          <div className="lm-cable-radio-header-chip">
            <div className="lm-cable-radio-header-art">
              {hit.imageUrl ? (
                <img src={hit.imageUrl} alt="" referrerPolicy="no-referrer" />
              ) : (
                <span className="lm-cable-radio-header-art--placeholder">📻</span>
              )}
              <span className="lm-cable-radio-header-live">{L.live}</span>
            </div>
            <div className="lm-cable-radio-header-copy">
              <p className="lm-cable-radio-header-name">{hit.title}</p>
              <p className="lm-cable-radio-header-meta">{hit.snippet}</p>
            </div>
            <div className="lm-cable-radio-header-vu" aria-hidden="true">
              {Array.from({ length: 7 }, (_, i) => (
                <span key={i} className="lm-cable-radio-header-vu-bar" style={{ animationDelay: `${i * 0.09}s` }} />
              ))}
            </div>
            <button
              type="button"
              className="lm-cable-radio-header-open"
              onClick={(e) => {
                e.stopPropagation();
                onOpenFull(activeIndex);
              }}
            >
              {L.open}
            </button>
          </div>
          <span className="lm-cable-radio-header-shine" aria-hidden="true" />
        </div>

        {nextHit ? (
          <button
            type="button"
            className="lm-cable-radio-header-peek lm-cable-radio-header-peek--next"
            onClick={() => onSelectIndex(nextIdx)}
            aria-label={nextHit.title}
          >
            {nextHit.imageUrl ? (
              <img src={nextHit.imageUrl} alt="" referrerPolicy="no-referrer" />
            ) : (
              <span>📻</span>
            )}
          </button>
        ) : null}
      </div>

      <div className="lm-cable-radio-header-dots" aria-hidden="true">
        {stations.slice(0, Math.min(8, total)).map((s, i) => (
          <span
            key={s.id}
            className={`lm-cable-radio-header-dot${i === activeIndex % 8 ? " is-active" : ""}`}
          />
        ))}
        {total > 8 ? <span className="lm-cable-radio-header-dot-more">+{total - 8}</span> : null}
      </div>

      {audioFocus && src ? (
        <HlsStreamPlayer
          key={`${hit.id}-${src}`}
          src={src}
          tag="audio"
          muted={muted}
          volume={volume}
          autoPlay
          multiView
          className="lm-cable-radio-audio"
          onCanPlay={onReady}
        />
      ) : null}
    </div>
  );
}
