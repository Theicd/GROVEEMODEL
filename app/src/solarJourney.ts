/** Multi-planet fly-by timeline — Earth → gap → Mars → (future worlds). */

export type SolarPlanetId = "earth" | "mars";

export type SolarSegment = {
  id: SolarPlanetId;
  /** Approach duration in seconds. */
  durationSec: number;
  /** Screen side: negative = left. */
  side: number;
};

/** Pause between planet fly-bys (deep space / warp). */
export const SOLAR_GAP_SEC = 45;

export const SOLAR_SEGMENTS: readonly SolarSegment[] = [
  { id: "earth", durationSec: 300, side: -1 },
  { id: "mars", durationSec: 225, side: -1 },
] as const;

/** Mars leg ~3.75 min. Total loop ~9.5 min before repeat. */
export function getSolarJourneyCycleSec(): number {
  const planets = SOLAR_SEGMENTS.reduce((sum, s) => sum + s.durationSec, 0);
  const gaps = SOLAR_GAP_SEC * Math.max(0, SOLAR_SEGMENTS.length - 1);
  return planets + gaps;
}

export type SolarJourneyState = {
  planetId: SolarPlanetId | null;
  /** 0–1 within active planet segment. */
  linear: number;
  segmentIndex: number;
  /** Elapsed within full journey loop. */
  loopSec: number;
};

export function getSolarJourneyState(elapsedSec: number): SolarJourneyState {
  const cycle = getSolarJourneyCycleSec();
  let t = ((elapsedSec % cycle) + cycle) % cycle;

  for (let i = 0; i < SOLAR_SEGMENTS.length; i++) {
    const seg = SOLAR_SEGMENTS[i];
    if (t < seg.durationSec) {
      return {
        planetId: seg.id,
        linear: t / seg.durationSec,
        segmentIndex: i,
        loopSec: t,
      };
    }
    t -= seg.durationSec;

    if (i < SOLAR_SEGMENTS.length - 1) {
      if (t < SOLAR_GAP_SEC) {
        return { planetId: null, linear: 0, segmentIndex: i, loopSec: t };
      }
      t -= SOLAR_GAP_SEC;
    }
  }

  return { planetId: null, linear: 0, segmentIndex: 0, loopSec: 0 };
}

/** @deprecated use SOLAR_SEGMENTS[0].durationSec */
export const EARTH_CYCLE_SEC = SOLAR_SEGMENTS[0].durationSec;

export const MARS_CYCLE_SEC = SOLAR_SEGMENTS.find((s) => s.id === "mars")?.durationSec ?? 225;
