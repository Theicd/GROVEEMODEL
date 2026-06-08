import type { EmotionScores } from "../vision-lab/core/types";
import { percent } from "../vision-lab/utils/geometry";

const EMOTIONS: Array<keyof Omit<EmotionScores, "dominant" | "dominantScore">> = [
  "happy",
  "neutral",
  "sad",
  "angry",
  "surprised",
  "fearful",
];

export function EmotionMeterPanel({
  emotion,
  compact = false,
  statusMessage,
  faceCount,
}: {
  emotion: EmotionScores | null;
  compact?: boolean;
  statusMessage?: string;
  faceCount?: number;
}) {
  if (!emotion) {
    if (!statusMessage) return null;
    return (
      <div className={`emotion-meter emotion-meter--idle ${compact ? "emotion-meter--compact" : ""}`}>
        <p className="emotion-meter-status">{statusMessage}</p>
        {faceCount !== undefined ? (
          <p className="emotion-meter-meta">Faces in frame: {faceCount}</p>
        ) : null}
      </div>
    );
  }

  return (
    <div className={`emotion-meter ${compact ? "emotion-meter--compact" : ""}`}>
      <div className="emotion-meter-head">
        <span className="emotion-meter-dominant">
          {emotion.dominant} ({percent(emotion.dominantScore)}%)
        </span>
        {faceCount !== undefined ? (
          <span className="emotion-meter-meta">{faceCount} face{faceCount === 1 ? "" : "s"}</span>
        ) : null}
      </div>
      <p className="emotion-meter-disclaimer">Estimate only — not clinical.</p>
      <div className="emotion-meter-bars">
        {EMOTIONS.map((key) => (
          <div key={key} className="emotion-meter-row">
            <div className="emotion-meter-labels">
              <span className="emotion-meter-name">{key}</span>
              <span className="emotion-meter-pct">{percent(emotion[key])}%</span>
            </div>
            <div className="emotion-meter-track">
              <div
                className="emotion-meter-fill"
                style={{ width: `${percent(emotion[key])}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
