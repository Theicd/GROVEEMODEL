type GlobeVisualProps = {
  size?: "xs" | "sm" | "md" | "lg";
  pulse?: boolean;
  /** `icon` — muted stroke like sidebar icons; `live` — cyan globe (dock / panel). */
  tone?: "live" | "icon";
};

export function GlobeVisual({ size = "md", pulse = true, tone = "live" }: GlobeVisualProps) {
  return (
    <div
      className={`globe-visual globe-visual--${size}${pulse ? " globe-visual--pulse" : ""}${tone === "icon" ? " globe-visual--icon" : ""}`}
      aria-hidden="true"
    >
      <div className="globe-visual-orbit" />
      <div className="globe-visual-orbit globe-visual-orbit--2" />
      <div className="globe-visual-sphere">
        <div className="globe-visual-atmosphere" />
        <div className="globe-visual-grid" />
      </div>
      <div className="globe-visual-satellite" />
    </div>
  );
}
