import { forwardRef } from "react";
import { moodLabelHe, type CharacterMood } from "./characterBrain";

type CameraPreviewProps = {
  active: boolean;
  observing?: boolean;
  mood?: CharacterMood;
  error?: string | null;
};

export const CameraPreview = forwardRef<HTMLVideoElement, CameraPreviewProps>(
  function CameraPreview({ active, observing, mood = "observing", error }, ref) {
    if (!active && !error) return null;

    const moodClass =
      mood === "excited"
        ? "camera-preview-badge--excited"
        : mood === "curious"
          ? "camera-preview-badge--curious"
          : observing
            ? "camera-preview-badge--live"
            : "";

    return (
      <div className={`camera-preview-wrap ${active ? "camera-preview-wrap--active" : ""}`}>
        {error ? (
          <p className="camera-preview-error" role="alert">
            {error}
          </p>
        ) : (
          <>
            <video ref={ref} className="camera-preview-video" aria-hidden="true" />
            <span className={`camera-preview-badge ${moodClass}`}>
              👁 {moodLabelHe(mood)}
            </span>
          </>
        )}
      </div>
    );
  },
);
