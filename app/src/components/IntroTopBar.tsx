import { GroveeLogoMark } from "../GroveeLogoMark";

type IntroTopBarProps = {
  webgpu: boolean;
};

export function IntroTopBar({ webgpu }: IntroTopBarProps) {
  return (
    <header className="intro-topbar" data-testid="intro-topbar" dir="rtl">
      <div className="intro-topbar__zone intro-topbar__zone--brand">
        <div className="intro-topbar__brand">
          <GroveeLogoMark size="xs" className="intro-topbar__logo" animated />
          <span className="intro-topbar__name" dir="ltr">
            GROVEE
          </span>
        </div>
      </div>

      <div className="intro-topbar__zone intro-topbar__zone--hud">
        <div className="intro-topbar__hud" dir="ltr" aria-label="סטטוס מערכת">
          <span className="intro-topbar__hud-cap intro-topbar__hud-cap--start" aria-hidden="true" />
          <div className={`intro-topbar__node ${webgpu ? "intro-topbar__node--live" : ""}`}>
            {webgpu ? <span className="intro-topbar__node-dot" aria-hidden="true" /> : null}
            <span className="intro-topbar__node-label">
              {webgpu ? "WebGPU פעיל" : "WebGPU לא זמין"}
            </span>
          </div>
          <span className="intro-topbar__hud-sep" aria-hidden="true" />
          <div className="intro-topbar__node intro-topbar__node--accent">
            <span className="intro-topbar__node-label">דפדפן מקומי</span>
          </div>
          <span className="intro-topbar__hud-sep" aria-hidden="true" />
          <div className="intro-topbar__node intro-topbar__node--dim">
            <span className="intro-topbar__node-label">GEMMA 4 E2B</span>
          </div>
          <span className="intro-topbar__hud-cap intro-topbar__hud-cap--end" aria-hidden="true" />
        </div>
      </div>

      <div className="intro-topbar__zone intro-topbar__zone--meta">
        <span className="intro-topbar__badge" title={import.meta.env.DEV ? "GROVEEMODEL dev — HAL space UI @ :5180" : undefined}>
          {import.meta.env.DEV ? "HAL·5180" : "v2026"}
        </span>
      </div>
    </header>
  );
}
