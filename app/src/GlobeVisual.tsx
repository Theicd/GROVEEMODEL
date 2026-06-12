type GlobeVisualProps = {
  size?: "sm" | "md" | "lg";
  pulse?: boolean;
};

export function GlobeVisual({ size = "md", pulse = true }: GlobeVisualProps) {
  return (
    <div className={`globe-visual globe-visual--${size}${pulse ? " globe-visual--pulse" : ""}`} aria-hidden="true">
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
