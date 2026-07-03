export type HurricaneIntensity = {
  category: number;
  color: number;
  colorSecondary: number;
  spinSpeed: number;
  particleCount: number;
  particleSize: number;
  maxRadius: number;
  eyeInner: number;
  eyeOuter: number;
  ringOuter: number;
  coreR: number;
  pickR: number;
  eyePulseHz: number;
  opacity: number;
};

const PROFILES: Record<number, Omit<HurricaneIntensity, "category">> = {
  1: {
    color: 0x33ccbb,
    colorSecondary: 0x228877,
    spinSpeed: 0.2,
    particleCount: 420,
    particleSize: 0.009,
    maxRadius: 0.06,
    eyeInner: 0.003,
    eyeOuter: 0.012,
    ringOuter: 0.038,
    coreR: 0.007,
    pickR: 0.07,
    eyePulseHz: 1.1,
    opacity: 0.55,
  },
  2: {
    color: 0x77dd44,
    colorSecondary: 0x559922,
    spinSpeed: 0.3,
    particleCount: 580,
    particleSize: 0.01,
    maxRadius: 0.075,
    eyeInner: 0.0035,
    eyeOuter: 0.014,
    ringOuter: 0.048,
    coreR: 0.008,
    pickR: 0.08,
    eyePulseHz: 1.35,
    opacity: 0.62,
  },
  3: {
    color: 0xffaa22,
    colorSecondary: 0xcc7700,
    spinSpeed: 0.44,
    particleCount: 720,
    particleSize: 0.011,
    maxRadius: 0.095,
    eyeInner: 0.004,
    eyeOuter: 0.017,
    ringOuter: 0.062,
    coreR: 0.0095,
    pickR: 0.095,
    eyePulseHz: 1.65,
    opacity: 0.72,
  },
  4: {
    color: 0xff4411,
    colorSecondary: 0xcc2200,
    spinSpeed: 0.62,
    particleCount: 900,
    particleSize: 0.013,
    maxRadius: 0.115,
    eyeInner: 0.0045,
    eyeOuter: 0.02,
    ringOuter: 0.078,
    coreR: 0.011,
    pickR: 0.11,
    eyePulseHz: 2.1,
    opacity: 0.82,
  },
  5: {
    color: 0xff1155,
    colorSecondary: 0xaa0033,
    spinSpeed: 0.88,
    particleCount: 1100,
    particleSize: 0.015,
    maxRadius: 0.14,
    eyeInner: 0.005,
    eyeOuter: 0.024,
    ringOuter: 0.095,
    coreR: 0.013,
    pickR: 0.13,
    eyePulseHz: 2.8,
    opacity: 0.92,
  },
};

export function parseWindKmh(severityText?: string): number | undefined {
  if (!severityText) return undefined;
  const m = severityText.match(/(\d+(?:\.\d+)?)\s*km\/h/i);
  return m ? Number(m[1]) : undefined;
}

/** Map sustained wind (km/h) to Saffir-Simpson-ish category. */
export function windToCategory(windKmh: number): number {
  if (windKmh >= 252) return 5;
  if (windKmh >= 209) return 4;
  if (windKmh >= 178) return 3;
  if (windKmh >= 154) return 2;
  if (windKmh >= 119) return 1;
  return 1;
}

export function getHurricaneIntensity(
  category?: number,
  windKmh?: number,
): HurricaneIntensity {
  let cat = Math.max(1, Math.min(5, Math.round(category ?? 2)));
  if (windKmh != null && Number.isFinite(windKmh)) {
    cat = Math.max(cat, windToCategory(windKmh));
  }
  const base = PROFILES[cat] ?? PROFILES[2];
  return { category: cat, ...base };
}

export function hurricaneColorCss(category?: number, severityText?: string): string {
  const { color } = getHurricaneIntensity(category, parseWindKmh(severityText));
  return `#${color.toString(16).padStart(6, "0")}`;
}
