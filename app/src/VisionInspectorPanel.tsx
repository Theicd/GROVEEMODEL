import type { RefObject } from "react";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";
import { clampInterval, intervalsFromMode } from "./vision-lab/core/schedule";
import type { PerformanceMode } from "./vision-lab/core/types";
import { VisionInspectorFeed } from "./VisionInspectorFeed";
import { VisionDashboard } from "./vision-dashboard/VisionDashboard";
import type { WorldInspectorSnapshot } from "./worldMemory";

const INTERVAL_FIELDS: Array<{ key: keyof PipelineConfig["sampleIntervals"]; label: string }> = [
  { key: "yolo", label: "Y" },
  { key: "pose", label: "P" },
  { key: "hands", label: "H" },
  { key: "face", label: "F" },
  { key: "emotion", label: "E" },
  { key: "uiUpdate", label: "UI" },
];

const MODEL_TOGGLES: Array<{ key: keyof PipelineConfig["toggles"]; label: string }> = [
  { key: "yolo", label: "YOLO" },
  { key: "pose", label: "Pose" },
  { key: "hands", label: "Hands" },
  { key: "face", label: "Face" },
  { key: "emotion", label: "Emo" },
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
      <div className="vi-panel" dir="ltr">
        <header className="vi-head">
          <div>
            <h2 id="vision-inspector-title">🔬 Vision Inspector</h2>
            <p className="vi-sub">
              {result.fps} FPS · {result.backend}
              {worldMemory?.lastVisionFrameAt ? ` · sync ${worldMemory.memoryAgeSec}s` : ""}
              {progress ? ` · ${progress}` : ""}
            </p>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="סגור">
            ×
          </button>
        </header>

        <div className="vi-video mirror">
          {cameraActive ? (
            <VisionInspectorFeed videoRef={videoRef} result={result} />
          ) : (
            <p className="vi-idle">הפעל מצלמה כדי לראות זיהוי חי</p>
          )}
        </div>

        {showDetectionCards ? (
          <div className="vi-cards">
            <VisionDashboard result={result} compact />
          </div>
        ) : null}

        <div className="vi-toolbar">
          <div className="vi-toolbar-group">
            <span className="vi-toolbar-label">Preset</span>
            {(["lite", "balanced", "full"] as PerformanceMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`vi-chip ${config.performanceMode === mode ? "active" : ""}`}
                onClick={() => applyPreset(mode)}
              >
                {mode}
              </button>
            ))}
          </div>

          <div className="vi-toolbar-group">
            <span className="vi-toolbar-label">Models</span>
            {MODEL_TOGGLES.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                className={`vi-chip ${config.toggles[key] ? "active" : ""}`}
                onClick={() => toggleModel(key)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="vi-toolbar-group vi-toolbar-intervals">
            <span className="vi-toolbar-label">ms</span>
            {INTERVAL_FIELDS.map(({ key, label }) => (
              <label key={key} className="vi-interval">
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
          </div>
        </div>

        {worldMemory ? (
          <footer className="vi-memory">
            <span>Obj: {worldMemory.objects.join(", ") || "—"}</span>
            <span>Person: {worldMemory.personPresent ? "yes" : "no"}</span>
            <span>Pose: {worldMemory.poseState}</span>
            <span>Gestures: {worldMemory.gestures.join(", ") || "—"}</span>
            {worldMemory.fingerStates.length ? (
              <span>
                Fingers: {worldMemory.fingerStates.map((f) => `${f.hand[0]}:${f.count}`).join(" ")}
              </span>
            ) : null}
          </footer>
        ) : null}
      </div>
    </div>
  );
}
