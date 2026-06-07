import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";
import { intervalsFromMode } from "./vision-lab/core/schedule";
import type { ModelToggles, PerformanceMode } from "./vision-lab/core/types";

const MODULE_LABELS: Record<keyof ModelToggles, string> = {
  yolo: "YOLO",
  pose: "Pose",
  hands: "Hands",
  face: "Face",
  emotion: "Emo",
  vlm: "VLM",
};

export function CameraVisionHud({
  result,
  config,
  progress,
  paused = false,
  onConfigChange,
}: {
  result: VisionResult;
  config: PipelineConfig;
  progress: string;
  paused?: boolean;
  onConfigChange?: (partial: Partial<PipelineConfig>) => void;
}) {
  const intervals = config.sampleIntervals;
  const uniqueObjects = [...new Map(result.objects.map((o) => [o.displayLabel, o])).values()];

  const toggleModule = (key: keyof ModelToggles) => {
    if (!onConfigChange || key === "vlm") return;
    onConfigChange({ toggles: { ...config.toggles, [key]: !config.toggles[key] } });
  };

  const setMode = (mode: PerformanceMode) => {
    onConfigChange?.({
      performanceMode: mode,
      sampleIntervals: intervalsFromMode(mode),
    });
  };

  return (
    <div className="camera-vision-hud" dir="ltr">
      <div className="camera-vision-hud-top">
        <span className="camera-vision-fps">
          {paused ? "⏸ paused" : `${result.fps} FPS`}
        </span>
        <span className="camera-vision-backend">{result.backend}</span>
        <select
          className="camera-vision-mode"
          value={config.performanceMode}
          onChange={(e) => setMode(e.target.value as PerformanceMode)}
          title="Performance preset (intervals)"
        >
          <option value="lite">lite</option>
          <option value="balanced">balanced</option>
          <option value="full">full</option>
        </select>
      </div>

      {progress ? <p className="camera-vision-progress">{progress}</p> : null}

      <div className="camera-vision-modules">
        {(Object.keys(config.toggles) as Array<keyof ModelToggles>).map((key) => (
          <button
            key={key}
            type="button"
            className={`camera-vision-module ${config.toggles[key] ? "on" : "off"}`}
            onClick={() => toggleModule(key)}
            disabled={!onConfigChange || key === "vlm"}
            title={key === "vlm" ? "VLM off in GROVEE (Gemma handles chat)" : `Toggle ${key}`}
          >
            {MODULE_LABELS[key]}
          </button>
        ))}
      </div>

      <p className="camera-vision-intervals" title="Sample intervals (ms)">
        Y{intervals.yolo} · P{intervals.pose} · H{intervals.hands} · F{intervals.face} · E
        {intervals.emotion} · UI{intervals.uiUpdate}
      </p>

      {uniqueObjects.length ? (
        <div className="camera-vision-objects">
          {uniqueObjects.slice(0, 5).map((obj) => (
            <span key={obj.displayLabel} className="camera-vision-chip">
              {obj.displayLabel} {Math.round(obj.confidence * 100)}%
            </span>
          ))}
        </div>
      ) : null}

      {result.hands.length ? (
        <div className="camera-vision-objects">
          {result.fingerStates.map((fs) => (
            <span key={fs.hand} className="camera-vision-chip camera-vision-chip--hands">
              {fs.hand} {fs.count}f
            </span>
          ))}
        </div>
      ) : null}

      {result.poseActions.length ? (
        <div className="camera-vision-objects">
          {result.poseActions.slice(0, 3).map((a) => (
            <span key={a.name} className="camera-vision-chip camera-vision-chip--pose">
              {a.name}
            </span>
          ))}
        </div>
      ) : null}

      {result.events.length ? (
        <p className="camera-vision-event">{result.events[0].name}</p>
      ) : result.sceneDescription ? (
        <p className="camera-vision-scene">{result.sceneDescription}</p>
      ) : null}
    </div>
  );
}
