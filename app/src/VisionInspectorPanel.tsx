import type { RefObject } from "react";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";
import { clampInterval, intervalsFromMode } from "./vision-lab/core/schedule";
import type { PerformanceMode } from "./vision-lab/core/types";
import { VisionInspectorFeed } from "./VisionInspectorFeed";
import { VisionDashboard } from "./vision-dashboard/VisionDashboard";
import type { WorldInspectorSnapshot } from "./worldMemory";

const INTERVAL_FIELDS: Array<{ key: keyof PipelineConfig["sampleIntervals"]; label: string }> = [
  { key: "yolo", label: "YOLO" },
  { key: "pose", label: "Pose" },
  { key: "hands", label: "Hands" },
  { key: "face", label: "Face" },
  { key: "emotion", label: "Emotion" },
  { key: "uiUpdate", label: "UI" },
];

const MODEL_TOGGLES: Array<{ key: keyof PipelineConfig["toggles"]; label: string }> = [
  { key: "yolo", label: "YOLO" },
  { key: "pose", label: "Pose" },
  { key: "hands", label: "Hands" },
  { key: "face", label: "Face" },
  { key: "emotion", label: "Emotion" },
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
  showDetectionCards = true,
  worldMemory,
}: {
  open: boolean;
  onClose: () => void;
  videoRef: RefObject<HTMLVideoElement | null>;
  result: VisionResult;
  config: PipelineConfig;
  onConfigChange: (partial: Partial<PipelineConfig>) => void;
  progress: string;
  cameraActive: boolean;
  showDetectionCards?: boolean;
  worldMemory?: WorldInspectorSnapshot | null;
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

  const toggleModel = (key: keyof PipelineConfig["toggles"]) => {
    onConfigChange({
      toggles: { ...config.toggles, [key]: !config.toggles[key] },
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
      <div className="activity-panel modal-box vision-inspector-panel vision-inspector-panel--wide" dir="ltr">
        <header className="activity-panel-head">
          <div>
            <h2 id="vision-inspector-title">🔬 Vision Inspector</h2>
            <p className="activity-panel-sub">
              YOLO · Pose · Hands · Face · Emotion — {result.fps} FPS · {result.backend}
              {worldMemory?.lastVisionFrameAt
                ? ` · memory ${worldMemory.memoryAgeSec}s ago`
                : ""}
            </p>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </header>

        <div className="vision-inspector-body vision-inspector-body--wide">
          <div className="vision-inspector-feed">
            {cameraActive ? (
              <div className="vision-inspector-video-wrap">
                <VisionInspectorFeed videoRef={videoRef} result={result} />
              </div>
            ) : (
              <p className="vision-inspector-idle">הפעל מצב מצלמה כדי לראות זיהוי חי</p>
            )}
            {progress ? <p className="vision-inspector-progress">{progress}</p> : null}

            {showDetectionCards ? (
              <div className="vision-inspector-cards">
                <VisionDashboard result={result} />
              </div>
            ) : null}
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
              <h3>Models</h3>
              <div className="vision-inspector-presets">
                {MODEL_TOGGLES.map(({ key, label }) => (
                  <button
                    key={key}
                    type="button"
                    className={config.toggles[key] ? "active" : ""}
                    onClick={() => toggleModel(key)}
                  >
                    {label}
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

            {worldMemory ? (
              <section>
                <h3>World Memory</h3>
                <ul className="vision-inspector-list">
                  <li>Objects: {worldMemory.objects.join(", ") || "(none)"}</li>
                  <li>Person: {worldMemory.personPresent ? "yes" : "no"}</li>
                  <li>Pose: {worldMemory.poseState}</li>
                  <li>Gestures: {worldMemory.gestures.join(", ") || "(none)"}</li>
                  <li>Holding: {worldMemory.holding.join(", ") || "(none)"}</li>
                  {worldMemory.fingerStates.length ? (
                    <li>
                      Fingers:{" "}
                      {worldMemory.fingerStates.map((f) => `${f.hand}:${f.count}`).join(", ")}
                    </li>
                  ) : null}
                  {worldMemory.faceSummary ? <li>Face: {worldMemory.faceSummary}</li> : null}
                  {worldMemory.emotionEstimate ? <li>Emotion est.: {worldMemory.emotionEstimate}</li> : null}
                </ul>
              </section>
            ) : null}

            <section>
              <h3>Scene</h3>
              <p className="vision-inspector-scene">{result.sceneDescription}</p>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}
