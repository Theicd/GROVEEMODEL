import { describe, expect, it, beforeEach, vi } from "vitest";
import { loadCableTunerSession, saveCableTunerSession, clearCableTunerSession } from "./cableTunerSession";

function mockSessionStorage() {
  const store = new Map<string, string>();
  const sessionStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => store.clear(),
    key: () => null,
    length: 0,
  };
  vi.stubGlobal("sessionStorage", sessionStorage);
  return store;
}

describe("cableTunerSession", () => {
  beforeEach(() => {
    mockSessionStorage();
    clearCableTunerSession();
  });

  it("round-trips tuner position in sessionStorage", () => {
    saveCableTunerSession({
      pageIndex: 12,
      quadSlots: [8, 9, 10, 11],
      rotationCursor: 4,
      selectedQuadSlot: 2,
    });
    expect(loadCableTunerSession()).toEqual({
      pageIndex: 12,
      quadSlots: [8, 9, 10, 11],
      rotationCursor: 4,
      selectedQuadSlot: 2,
    });
  });
});
