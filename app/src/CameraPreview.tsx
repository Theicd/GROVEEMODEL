import { forwardRef, useCallback, useRef, type MutableRefObject, type Ref } from "react";
import { moodLabelHe, type CharacterMood } from "./characterBrain";
import { VisionDetectionOverlay } from "./VisionDetectionOverlay";
import { CameraVisionHud } from "./CameraVisionHud";
import { HalPerceptionHud } from "./HalPerceptionHud";
import type { HalMoodState } from "./vision2/halMoodEngine";
import type { EntityProfile } from "./vision2/entityProfile";
import type { ConsciousnessLayer, InterpretationLayer } from "./vision2/types";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";

function mergeRefs<T>(...refs: Array<Ref<T> | undefined>): (value: T | null) => void {
  return (value) => {
    for (const ref of refs) {
      if (!ref) continue;
      if (typeof ref === "function") ref(value);
      else (ref as MutableRefObject<T | null>).current = value;
    }
  };
}

type CameraPreviewProps = {
  active: boolean;
  observing?: boolean;
  mood?: CharacterMood;
  hal?: HalMoodState | null;
  interpretation?: InterpretationLayer | null;
  consciousness?: ConsciousnessLayer | null;
  entity?: EntityProfile | null;
  cameraStatus?: string;
  error?: string | null;
  visionResult?: VisionResult | null;
  pipelineConfig?: PipelineConfig | null;
  pipelineProgress?: string;
  visionPaused?: boolean;
  onVideoReady?: (video: HTMLVideoElement) => void;
  onPipelineConfigChange?: (partial: Partial<PipelineConfig>) => void;
  /** Desktop right panel vs inline above composer (mobile). */
  variant?: "inline" | "panel";
};

export const CameraPreview = forwardRef<HTMLVideoElement, CameraPreviewProps>(
  function CameraPreview(
    {
      active,
      observing,
      mood = "observing",
      hal = null,
      interpretation = null,
      consciousness = null,
      entity = null,
      cameraStatus = "",
      error,
      visionResult,
      pipelineConfig,
      pipelineProgress = "",
      visionPaused = false,
      onVideoReady,
      onPipelineConfigChange,
      variant = "inline",
    },
    ref,
  ) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const notifiedRef = useRef(false);

    const handleVideoRef = useCallback(
      (el: HTMLVideoElement | null) => {
        mergeRefs(ref, videoRef)(el);
        // Match browser-vision-lab: attach stream in parent, do not wait for loadeddata first.
        if (el && active && onVideoReady && !notifiedRef.current) {
          notifiedRef.current = true;
          onVideoReady(el);
        }
        if (!el) notifiedRef.current = false;
      },
      [ref, onVideoReady, active],
    );

    if (!active && !error) return null;

    const moodClass =
      mood === "excited"
        ? "camera-preview-badge--excited"
        : mood === "curious"
          ? "camera-preview-badge--curious"
          : observing
            ? "camera-preview-badge--live"
            : "";

    const showOverlay = active && !!visionResult;
    const showHud = showOverlay && !!pipelineConfig;

    return (
      <div
        className={`camera-preview-wrap ${active ? "camera-preview-wrap--active" : ""} ${variant === "panel" ? "camera-preview-wrap--panel" : ""}`}
      >
        {error ? (
          <p className="camera-preview-error" role="alert">
            {error}
          </p>
        ) : (
          <div className="camera-preview-column">
            <div className="camera-preview-stage mirror">
              <video
                ref={handleVideoRef}
                className="camera-preview-video"
                aria-hidden="true"
                playsInline
                muted
              />
              {showOverlay ? (
                <VisionDetectionOverlay
                  videoRef={videoRef}
                  result={visionResult!}
                  compact
                />
              ) : null}
            </div>

            {showHud ? (
              <CameraVisionHud
                result={visionResult!}
                config={pipelineConfig!}
                progress={pipelineProgress}
                paused={visionPaused}
                onConfigChange={onPipelineConfigChange}
              />
            ) : null}

            <HalPerceptionHud
              hal={hal}
              interpretation={interpretation}
              consciousness={consciousness}
              entity={entity}
              mood={mood}
              cameraStatus={cameraStatus}
            />

            <span className={`camera-preview-badge ${moodClass}`}>
              👁 {moodLabelHe(mood)}
            </span>
          </div>
        )}
      </div>
    );
  },
);
