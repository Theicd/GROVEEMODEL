import type { HalMoodState } from "./vision2/halMoodEngine";

import type { EntityProfile } from "./vision2/entityProfile";
import type { ConsciousnessLayer, InterpretationLayer } from "./vision2/types";

import { activityLabelHe } from "./vision2/interpretation/eventFusion";

import { SOUL_LABEL_HE } from "./vision2/consciousness/types";

import { moodLabelHe, type CharacterMood } from "./characterBrain";



type HalPerceptionHudProps = {

  hal: HalMoodState | null;

  interpretation?: InterpretationLayer | null;

  consciousness?: ConsciousnessLayer | null;
  entity?: EntityProfile | null;

  mood: CharacterMood;

  cameraStatus?: string;

};



export function HalPerceptionHud({

  hal,

  interpretation,

  consciousness,
  entity,
  mood,
  cameraStatus,
}: HalPerceptionHudProps) {

  if (!hal) {

    return (

      <div className="hal-perception-hud" dir="rtl">

        <span className="hal-perception-hud__chip hal-perception-hud__chip--muted">

          HAL · מאתחל…

        </span>

      </div>

    );

  }



  const soul = consciousness?.soul;

  const personLabel = consciousness

    ? `${SOUL_LABEL_HE[soul ?? "VOID_IDLE"]} ${Math.round((consciousness.confidence ?? 0) * 100)}%`

    : hal.personPresent

      ? "אדם בפריים ✓"

      : "אין אדם מאושר";

  const personClass =

    consciousness?.personStable || (!consciousness && hal.personPresent)

      ? "hal-perception-hud__chip--person-yes"

      : consciousness?.soul === "PHANTOM_DETECTION" || consciousness?.soul === "PRESENCE_FORMING"

        ? "hal-perception-hud__chip--person-weak"

        : "hal-perception-hud__chip--person-no";



  return (

    <div className="hal-perception-hud" dir="rtl">

      <span className={`hal-perception-hud__chip hal-perception-hud__chip--mood hal-perception-hud__mood--${mood}`}>

        🧠 {moodLabelHe(mood)} · {hal.tone}

      </span>

      {consciousness ? (

        <span className="hal-perception-hud__chip hal-perception-hud__chip--soul" title={consciousness.interpretation}>

          🧊 {consciousness.soul.replace(/_/g, " ")}

          {consciousness.evolution !== consciousness.soul.replace(/_/g, " ")

            ? ` · ${consciousness.evolution.split(" → ").slice(-3).join(" → ")}`

            : ""}

        </span>

      ) : null}

      {interpretation ? (

        <span className="hal-perception-hud__chip hal-perception-hud__chip--scene">

          🎬 {activityLabelHe(interpretation.sceneState.activity as import("./vision2/interpretation/sceneStateEngine").SceneActivity)} · {interpretation.sceneState.stability}

        </span>

      ) : null}

      {interpretation?.metaEvents[0] ? (

        <span

          className="hal-perception-hud__chip hal-perception-hud__chip--meta"

          title={interpretation.metaEvents[0].meaning}

        >

          ⚡ {interpretation.metaEvents[0].type.replace(/_/g, " ")}

        </span>

      ) : null}

      <span className={`hal-perception-hud__chip ${personClass}`}>👤 {personLabel}</span>

      {entity && entity.faceObservations > 0 ? (
        <span className="hal-perception-hud__chip hal-perception-hud__chip--entity" title={`${entity.emotion ?? ""} · ${entity.engagement}`}>
          🪪 {entity.segment === "child" ? "ילד/ה" : entity.segment === "teen" ? "נער/ה" : entity.segment === "adult" ? "מבוגר/ת" : "?"}
          {entity.ageEstimate ? ` ~${entity.ageEstimate}` : ""}
          {entity.gender !== "unknown" ? ` · ${entity.gender === "male" ? "♂" : "♀"}` : ""}
          {` · ${entity.engagement}`}
        </span>
      ) : null}
      {consciousness && consciousness.rawDetected !== consciousness.personStable ? (
        <span className="hal-perception-hud__chip hal-perception-hud__chip--muted" title="רעש חיישן — לא אמת">
          📡 {consciousness.rawDetected ? "חיישן: זוהה" : "חיישן: ריק"} · לא יציב
        </span>
      ) : null}

      <span className="hal-perception-hud__chip hal-perception-hud__chip--situation">

        📍 {hal.situationPrimary} {Math.round(hal.situationConfidence * 100)}%

      </span>

      {hal.sceneLabel ? (

        <span className="hal-perception-hud__chip hal-perception-hud__chip--scene" title={hal.interpretation ?? ""}>

          🎬 {hal.sceneLabel}

        </span>

      ) : null}

      {cameraStatus ? (

        <span className="hal-perception-hud__chip hal-perception-hud__chip--status">{cameraStatus}</span>

      ) : null}

    </div>

  );

}

