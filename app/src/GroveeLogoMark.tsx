type GroveeLogoMarkProps = {
  size?: "xs" | "sm" | "md" | "lg";
  /** Default true — set false for static avatars (e.g. chat messages). */
  animated?: boolean;
  className?: string;
};

/** Animated ring logo from the intro screen — reusable in sidebar / header. */
export function GroveeLogoMark({ size = "md", animated = true, className = "" }: GroveeLogoMarkProps) {
  return (
    <div
      className={`grovee-logo-mark grovee-logo-mark--${size}${animated ? "" : " grovee-logo-mark--static"} ${className}`.trim()}
      aria-hidden="true"
    >
      <div className="ring r1" />
      <div className="ring r2" />
      <div className="ring r3" />
    </div>
  );
}
