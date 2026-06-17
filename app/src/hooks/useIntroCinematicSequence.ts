import { useEffect, useState } from "react";

import { CINEMATIC_TIMELINE, type IntroCinematicStage } from "../introCinematicLines";

export function useIntroCinematicSequence(active: boolean) {
  const [stage, setStage] = useState<IntroCinematicStage>("idle");

  useEffect(() => {
    if (!active) {
      setStage("idle");
      return;
    }

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setStage("action");
      return;
    }

    setStage("idle");
    const timers = CINEMATIC_TIMELINE.map(({ stage: next, at }) =>
      window.setTimeout(() => setStage(next), at),
    );

    return () => timers.forEach((id) => window.clearTimeout(id));
  }, [active]);

  return { stage };
}
