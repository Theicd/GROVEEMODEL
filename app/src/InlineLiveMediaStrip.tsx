import { LiveMediaResultsGrid } from "./searchResults/LiveMediaResultsGrid";
import type { ChatUiLanguage } from "./ui/useUiLanguage";
import type { UnifiedSearchHit } from "./searchResults/types";

type Props = {
  hits: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  mode: "livetv" | "radio";
  sportsPackage?: boolean;
  onOpenSportsPackage?: () => void;
};

export function InlineLiveMediaStrip({
  hits,
  uiLang,
  mode,
  sportsPackage,
  onOpenSportsPackage,
}: Props) {
  if (!hits.length) return null;
  const rtl = uiLang === "he";
  return (
    <div className="inline-live-media-strip" dir={rtl ? "rtl" : "ltr"}>
      <div className="inline-live-media-head">
        <span>{mode === "radio" ? "📻 רדיו חי" : "📺 ערוצים חיים"}</span>
        {sportsPackage && onOpenSportsPackage ? (
          <button type="button" className="inline-live-sports-btn" onClick={onOpenSportsPackage}>
            {rtl ? "חבילת ספורט →" : "Sports package →"}
          </button>
        ) : null}
      </div>
      <LiveMediaResultsGrid hits={hits.slice(0, 8)} uiLang={uiLang} mode={mode} />
    </div>
  );
}
