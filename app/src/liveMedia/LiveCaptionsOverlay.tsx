type Props = {
  original: string;
  translated: string;
  visible: boolean;
};

export function LiveCaptionsOverlay({ original, translated, visible }: Props) {
  if (!visible || (!original && !translated)) return null;
  const showTranslation =
    Boolean(translated) && translated !== "…" && translated.trim() !== original.trim();
  return (
    <div className="lm-cable-captions" aria-live="polite">
      {original ? <p className="lm-cable-captions__orig">{original}</p> : null}
      {showTranslation ? <p className="lm-cable-captions__trans">{translated}</p> : null}
    </div>
  );
}
