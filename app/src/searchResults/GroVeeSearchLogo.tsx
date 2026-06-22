/** Daily-rotating GroVee wordmark — Google Doodle–style accents for the search landing. */

export type GroVeeDoodleId = "classic" | "waves" | "news" | "orbit" | "pulse" | "ships";

const DOODLE_CYCLE: GroVeeDoodleId[] = ["classic", "waves", "news", "orbit", "pulse", "ships"];

const LETTER_COLORS = ["#4285f4", "#ea4335", "#fbbc04", "#34a853", "#00f3ff", "#10a37f"];

export const pickGroVeeDoodleId = (date = new Date()): GroVeeDoodleId => {
  const day = Math.floor(date.getTime() / 86_400_000);
  return DOODLE_CYCLE[day % DOODLE_CYCLE.length] ?? "classic";
};

type Props = {
  doodle?: GroVeeDoodleId;
  compact?: boolean;
  className?: string;
};

export function GroVeeSearchLogo({ doodle, compact = false, className = "" }: Props) {
  const id = doodle ?? pickGroVeeDoodleId();

  return (
    <div
      className={`serp-grovee-logo serp-grovee-logo--${id}${compact ? " serp-grovee-logo--compact" : ""} ${className}`.trim()}
      role="img"
      aria-label="GroVee"
    >
      <span className="serp-grovee-logo-word" aria-hidden="true">
        {"GroVee".split("").map((ch, i) => (
          <span key={`${ch}-${i}`} className="serp-grovee-logo-letter" style={{ color: LETTER_COLORS[i] }}>
            {ch}
          </span>
        ))}
      </span>
      <span className="serp-grovee-logo-accent" aria-hidden="true" />
    </div>
  );
}
