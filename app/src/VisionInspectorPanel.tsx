import type { RefObject } from "react";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";
import { clampInterval, intervalsFromMode } from "./vision-lab/core/schedule";
import type { PerformanceMode } from "./vision-lab/core/types";
import { VisionInspectorFeed } from "./VisionInspectorFeed";

const INTERVAL_FIELDS: Array<{ key: keyof PipelineConfig["sampleIntervals"]; label: string }> = [
  { key: "yolo", label: "YOLO" },
  { key: "pose", label: "Pose" },
  { key: "hands", label: "Hands" },
  { key: "face", label: "Face" },
  { key: "emotion", label: "Emotion" },
  { key: "uiUpdate", label: "UI" },
];

export function VisionInspectorPanel({
  open,
  onClose,
  videoRef,
  result,
  config,
  onConfigChange,
  progress,
  cameraActive,
}: {
  open: boolean;
  onClose: () => void;
  videoRef: RefObject<HTMLVideoElement | null>;
  result: VisionResult;
  config: PipelineConfig;
  onConfigChange: (partial: Partial<PipelineConfig>) => void;
  progress: string;
  cameraActive: boolean;
}) {
  if (!open) return null;

  const applyPreset = (mode: PerformanceMode) => {
    onConfigChange({
      performanceMode: mode,
      sampleIntervals: intervalsFromMode(mode),
    });
  };

  const setInterval = (key: keyof PipelineConfig["sampleIntervals"], value: string) => {
    onConfigChange({
      sampleIntervals: {
        ...config.sampleIntervals,
        [key]: clampInterval(Number(value)),
      },
    });
  };

  return (
    <div
      className="activity-overlay modal vision-inspector-overlay-wrap"
      role="dialog"
      aria-modal="true"
      aria-labelledby="vision-inspector-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="activity-panel modal-box vision-inspector-panel" dir="ltr">
        <header className="activity-panel-head">
          <div>
            <h2 id="vision-inspector-title">🔬 Vision Inspector</h2>
            <p className="activity-panel-sub">
              YOLO · Pose · Hands · Face · Emotion — {result.fps} FPS · {result.backend}
            </p>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </header>

        <div className="vision-inspector-body">
          <div className="vision-inspector-feed">
            {cameraActive ? (
              <div className="vision-inspector-video-wrap">
                <VisionInspectorFeed videoRef={videoRef} result={result} />
              </div>
            ) : (
              <p className="vision-inspector-idle">הפעל מצב מצלמה כדי לראות זיהוי חי</p>
            )}
            {progress ? <p className="vision-inspector-progress">{progress}</p> : null}
          </div>

          <aside className="vision-inspector-sidebar" dir="ltr">
            <section>
              <h3>Presets</h3>
              <div className="vision-inspector-presets">
                {(["lite", "balanced", "full"] as PerformanceMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={config.performanceMode === mode ? "active" : ""}
                    onClick={() => applyPreset(mode)}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </section>

            <section>
              <h3>Intervals (ms)</h3>
              {INTERVAL_FIELDS.map(({ key, label }) => (
                <label key={key} className="vision-inspector-interval">
                  <span>{label}</span>
                  <input
                    type="number"
                    min={100}
                    max={60000}
                    step={50}
                    value={config.sampleIntervals[key]}
                    onChange={(e) => setInterval(key, e.target.value)}
                  />
                </label>
              ))}
            </section>

            <section>
              <h3>Scene</h3>
              <p className="vision-inspector-scene">{result.sceneDescription}</p>
            </section>

            {result.events.length ? (
              <section>
                <h3>Events</h3>
                <ul className="vision-inspector-list">
                  {result.events.map((e) => (
                    <li key={e.name}>
                      {e.name} ({Math.round(e.confidence * 100)}%)
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}

            {result.bodyLanguage.length ? (
              <section>
                <h3>Body language</h3>
                <ul className="vision-inspector-list">
                  {result.bodyLanguage.slice(0, 5).map((c) => (
                    <li key={c.signal}>
                      {c.signal}: {c.meaning}
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}

            {result.emotion ? (
              <section>
                <h3>Expression (estimate)</h3>
                <p className="vision-inspector-emotion-disclaimer">
                  Estimate only — not a clinical reading.
                </p>
                <p>
                  {result.emotion.dominant} ({Math.round(result.emotion.dominantScore * 100)}%)
                </p>
              </section>
            ) : null}
          </aside>
        </div>
      </div>
    </div>
  );
}
