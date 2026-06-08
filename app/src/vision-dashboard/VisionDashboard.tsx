import type { VisionResult } from "../vision-lab/core/types";
import { VisionCard } from "./VisionCard";

function Row({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="vision-dash-row">
      <span>{label}</span>
      <span className={className ?? "vision-dash-value"}>{value}</span>
    </div>
  );
}

export function VisionDashboard({ result }: { result: VisionResult }) {
  const uniqueObjects = [...new Map(result.objects.map((o) => [o.displayLabel, o])).values()];

  return (
    <div className="vision-dash-grid">
      <VisionCard title="Objects" empty={uniqueObjects.length === 0}>
        {uniqueObjects.map((obj) => (
          <Row key={obj.displayLabel} label={obj.displayLabel} value={`${Math.round(obj.confidence * 100)}%`} />
        ))}
      </VisionCard>

      <VisionCard title="Pose" empty={result.poseActions.length === 0}>
        {result.poseActions.slice(0, 6).map((a) => (
          <Row key={a.name} label={a.name} value={`${Math.round(a.confidence * 100)}%`} className="vision-dash-value pose" />
        ))}
      </VisionCard>

      <VisionCard
        title="Hands"
        empty={result.fingerStates.length === 0 && result.staticGestures.length === 0}
      >
        {result.fingerStates.map((fs) => (
          <div key={fs.hand} className="vision-dash-hand-block">
            <div className="vision-dash-hand-title">
              {fs.hand} — {fs.count} finger{fs.count === 1 ? "" : "s"}
            </div>
            <div className="vision-dash-fingers">
              {Object.entries(fs.fingers).map(([finger, state]) => (
                <span key={finger}>
                  {finger}: {state}
                </span>
              ))}
            </div>
          </div>
        ))}
        {result.staticGestures.slice(0, 4).map((g, i) => (
          <div key={`${g.name}-${i}`} className="vision-dash-gesture-static">
            {g.hand}: {g.name}
          </div>
        ))}
      </VisionCard>

      <VisionCard
        title="Gestures"
        empty={result.staticGestures.length === 0 && result.motionGestures.length === 0}
      >
        {result.motionGestures.map((g, i) => (
          <Row key={`m-${g.name}-${i}`} label={g.name} value={`${Math.round(g.confidence * 100)}%`} className="vision-dash-value motion" />
        ))}
        {result.staticGestures.map((g, i) => (
          <Row key={`s-${g.name}-${i}`} label={g.name} value={`${Math.round(g.confidence * 100)}%`} className="vision-dash-value sign" />
        ))}
      </VisionCard>

      <VisionCard title="Body Language" empty={result.bodyLanguage.length === 0}>
        {result.bodyLanguage.map((cue) => (
          <div key={`${cue.category}-${cue.signal}`} className="vision-dash-cue">
            <div className="vision-dash-cue-head">
              <span>{cue.signal}</span>
              <span className={`vision-dash-cat vision-dash-cat--${cue.category}`}>{cue.category}</span>
            </div>
            <div className="vision-dash-cue-meaning">{cue.meaning}</div>
            <div className="vision-dash-cue-conf">{Math.round(cue.confidence * 100)}%</div>
          </div>
        ))}
      </VisionCard>

      <VisionCard
        title="Actions & Events"
        empty={result.interactions.length === 0 && result.events.length === 0}
      >
        {result.interactions.map((i, idx) => (
          <Row key={`int-${i.name}-${idx}`} label={i.name} value={`${Math.round(i.confidence * 100)}%`} className="vision-dash-value event" />
        ))}
        {result.events.map((e, idx) => (
          <Row key={`ev-${e.name}-${idx}`} label={e.name} value={`${Math.round(e.confidence * 100)}%`} className="vision-dash-value event" />
        ))}
      </VisionCard>

      <VisionCard title="Face" empty={result.faces.length === 0}>
        <div className="vision-dash-hand-title">Count: {result.faces.length}</div>
        {result.faces.map((face) => (
          <div key={face.id} className="vision-dash-face-block">
            <div>Face #{face.id}</div>
            <div>Age est.: {Math.round(face.estimatedAge)}</div>
            <div>Gender est.: {face.estimatedGender}</div>
            <div>Gaze: {face.gazeDirection}</div>
          </div>
        ))}
      </VisionCard>

      <VisionCard title="Emotion" empty={!result.emotion}>
        {result.emotion ? (
          <>
            <p className="vision-inspector-emotion-disclaimer">Estimate only — not clinical.</p>
            <Row
              label={result.emotion.dominant}
              value={`${Math.round(result.emotion.dominantScore * 100)}%`}
              className="vision-dash-value emotion"
            />
          </>
        ) : null}
      </VisionCard>

      <VisionCard title="Environment" empty={result.environment === "Unknown"}>
        <div className="vision-dash-env">{result.environment}</div>
      </VisionCard>

      <VisionCard title="Scene" empty={!result.sceneDescription && !result.vlmDescription}>
        {result.sceneDescription ? <p className="vision-dash-scene">{result.sceneDescription}</p> : null}
        {result.vlmDescription ? (
          <p className="vision-dash-scene vision-dash-scene--vlm">{result.vlmDescription}</p>
        ) : null}
      </VisionCard>
    </div>
  );
}
