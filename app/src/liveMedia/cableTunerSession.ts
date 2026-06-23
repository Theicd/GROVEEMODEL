const SESSION_KEY = "grovee-cable-tuner-session";

export type CableTunerSession = {
  pageIndex: number;
  quadSlots: number[];
  rotationCursor: number;
  selectedQuadSlot: number;
};

export function loadCableTunerSession(): CableTunerSession | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CableTunerSession;
    if (!Number.isFinite(parsed.pageIndex)) return null;
    if (!Array.isArray(parsed.quadSlots) || parsed.quadSlots.length !== 4) return null;
    return {
      pageIndex: Math.max(0, Math.floor(parsed.pageIndex)),
      quadSlots: parsed.quadSlots.map((n) => Math.max(0, Math.floor(Number(n) || 0))),
      rotationCursor: Math.max(0, Math.floor(parsed.rotationCursor ?? 0)),
      selectedQuadSlot: Math.max(0, Math.min(3, Math.floor(parsed.selectedQuadSlot ?? 0))),
    };
  } catch {
    return null;
  }
}

export function saveCableTunerSession(state: CableTunerSession): void {
  try {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(state));
  } catch {
    /* quota / private mode */
  }
}

export function clearCableTunerSession(): void {
  try {
    sessionStorage.removeItem(SESSION_KEY);
  } catch {
    /* ignore */
  }
}
