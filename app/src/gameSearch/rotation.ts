const SESSION_KEY = "grovee-game-rotation-v2";



type RotationState = Record<string, number>;



const readState = (): RotationState => {

  try {

    if (typeof sessionStorage === "undefined") return {};

    const raw = sessionStorage.getItem(SESSION_KEY);

    return raw ? (JSON.parse(raw) as RotationState) : {};

  } catch {

    return {};

  }

};



const writeState = (state: RotationState): void => {

  try {

    if (typeof sessionStorage !== "undefined") {

      sessionStorage.setItem(SESSION_KEY, JSON.stringify(state));

    }

  } catch {

    /* ignore quota */

  }

};



/** Monotonic page counter per key — new page each call (wraps at maxPages). */

export const nextRotationPage = (key: string, maxPages: number): number => {

  const cap = Math.max(1, maxPages);

  const state = readState();

  const prev = state[key] ?? 0;

  const page = (prev % cap) + 1;

  state[key] = page;

  writeState(state);

  return page;

};



/** Random page biased toward unexplored offsets (session-aware). */

export const pickRandomPage = (key: string, maxPages: number): number => {

  const cap = Math.max(1, maxPages);

  const rotated = nextRotationPage(key, cap);

  const jitter = Math.floor(Math.random() * Math.min(5, cap));

  return ((rotated + jitter - 1) % cap) + 1;

};



export const resetRotationSession = (): void => {

  try {

    sessionStorage?.removeItem(SESSION_KEY);

  } catch {

    /* ignore */

  }

};

