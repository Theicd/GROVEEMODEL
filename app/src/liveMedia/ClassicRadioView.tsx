import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "../searchResults/types";
import { HlsStreamPlayer } from "../searchResults/HlsStreamPlayer";

type Props = {
  hit: UnifiedSearchHit;
  stationIndex: number;
  stationTotal: number;
  regionLabel: string;
  uiLang: ChatUiLanguage;
  muted: boolean;
  volume: number;
};

export function ClassicRadioView({
  hit,
  stationIndex,
  stationTotal,
  regionLabel,
  uiLang,
  muted,
  volume,
}: Props) {
  const rtl = uiLang === "he";
  const src = hit.mediaPlayUrl || hit.url || "";
  const L =
    uiLang === "he"
      ? {
          radio: "רדיו",
          region: "אזור",
          station: "תחנה",
          live: "שידור חי",
        }
      : {
          radio: "Radio",
          region: "Region",
          station: "Station",
          live: "Live",
        };

  return (
    <div className="lm-cable-classic-radio" dir={rtl ? "rtl" : "ltr"}>
      <div className="lm-cable-classic-radio-cabinet">
        <div className="lm-cable-classic-radio-grill" aria-hidden="true" />
        <div className="lm-cable-classic-radio-face">
          <div className="lm-cable-classic-radio-brand">
            <span className="lm-cable-classic-radio-brand-mark">GROVEE</span>
            <span className="lm-cable-classic-radio-brand-sub">{L.radio}</span>
          </div>

          <div className="lm-cable-classic-radio-tuner">
            <div className="lm-cable-classic-radio-window">
              {hit.imageUrl ? (
                <img className="lm-cable-classic-radio-art" src={hit.imageUrl} alt="" referrerPolicy="no-referrer" />
              ) : (
                <div className="lm-cable-classic-radio-art lm-cable-classic-radio-art--placeholder">📻</div>
              )}
              <span className="lm-cable-classic-radio-live">{L.live}</span>
            </div>
            <div className="lm-cable-classic-radio-readout">
              <p className="lm-cable-classic-radio-title">{hit.title}</p>
              <p className="lm-cable-classic-radio-snippet">{hit.snippet}</p>
            </div>
          </div>

          <div className="lm-cable-classic-radio-meters" aria-hidden="true">
            {Array.from({ length: 8 }, (_, i) => (
              <span key={i} className="lm-cable-classic-radio-meter-bar" style={{ animationDelay: `${i * 0.08}s` }} />
            ))}
          </div>

          <div className="lm-cable-classic-radio-osd">
            <span>
              {L.region}: {regionLabel}
            </span>
            <span>
              {L.station} {stationIndex + 1}/{stationTotal}
            </span>
          </div>
        </div>
      </div>

      <HlsStreamPlayer
        key={`${hit.id}-${src}`}
        src={src}
        tag="audio"
        muted={muted}
        volume={volume}
        autoPlay
        className="lm-cable-radio-audio"
      />
    </div>
  );
}
