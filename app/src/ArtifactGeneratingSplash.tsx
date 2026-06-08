import { IntroCanvas } from "./IntroCanvas";

type ArtifactGeneratingSplashProps = {
  tokenCount?: number;
};

export function ArtifactGeneratingSplash({ tokenCount = 0 }: ArtifactGeneratingSplashProps) {
  return (
    <div className="artifact-generating-splash" aria-live="polite" aria-busy="true">
      <IntroCanvas contained />
      <div className="artifact-generating-content">
        <div className="core-visual core-visual--compact" aria-hidden="true">
          <div className="ring r1" />
          <div className="ring r2" />
          <div className="ring r3" />
        </div>
        <p className="artifact-generating-title">
          GENERATING CODE
          <span className="artifact-generating-dots" aria-hidden="true">
            ....
          </span>
        </p>
        {tokenCount > 0 ? (
          <div className="artifact-generating-tokens" dir="ltr" aria-live="polite">
            <span className="artifact-generating-tokens-num" key={tokenCount}>
              {tokenCount.toLocaleString()}
            </span>
            <span className="artifact-generating-tokens-label">TOKENS</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
