import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  OFFLINE_PACK_COMPONENTS,
  OFFLINE_PACK_LOCAL_STORAGE_KEY,
  clearOfflinePackCompletedAt,
  formatOfflinePackCompletedAt,
  initialOfflinePackState,
  mapImagePhaseProgress,
  mapModelsPhaseProgress,
  offlinePackComponentLabel,
  offlinePackTotalBytes,
  offlinePackTotalLabel,
  readOfflinePackCompletedAt,
  writeOfflinePackCompletedAt,
} from "./offlinePack";

const installLocalStorage = () => {
  const store = new Map<string, string>();
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (k: string) => (store.has(k) ? (store.get(k) as string) : null),
      setItem: (k: string, v: string) => void store.set(k, String(v)),
      removeItem: (k: string) => void store.delete(k),
      clear: () => store.clear(),
      key: (i: number) => Array.from(store.keys())[i] ?? null,
      get length() {
        return store.size;
      },
    } as Storage,
  });
};

beforeEach(() => installLocalStorage());
afterEach(() => clearOfflinePackCompletedAt());

describe("offlinePack — registry & sizes", () => {
  it("includes the four expected components in stable order", () => {
    expect(OFFLINE_PACK_COMPONENTS.map((c) => c.id)).toEqual(["chat", "code", "caption", "image"]);
  });

  it("total size is roughly 5 GB and label uses GB", () => {
    const total = offlinePackTotalBytes();
    expect(total).toBeGreaterThan(4 * 1024 ** 3);
    expect(total).toBeLessThan(7 * 1024 ** 3);
    expect(offlinePackTotalLabel()).toMatch(/GB$/);
  });

  it("component label embeds the human-readable size", () => {
    const sd = OFFLINE_PACK_COMPONENTS.find((c) => c.id === "image")!;
    expect(offlinePackComponentLabel(sd)).toMatch(/SD-Turbo/);
    expect(offlinePackComponentLabel(sd)).toMatch(/GB/);
  });
});

describe("offlinePack — progress mapping", () => {
  it("maps the models phase into the first 70% of the bar (clamped)", () => {
    expect(mapModelsPhaseProgress(0)).toBe(0);
    expect(mapModelsPhaseProgress(50)).toBe(35);
    expect(mapModelsPhaseProgress(100)).toBe(70);
    expect(mapModelsPhaseProgress(150)).toBe(70);
    expect(mapModelsPhaseProgress(-10)).toBe(0);
  });

  it("maps the image phase into the last 30% of the bar by parsing 'NN%'", () => {
    expect(mapImagePhaseProgress("Local image: 0% downloading")).toBe(70);
    expect(mapImagePhaseProgress("Local image: 50% unet")).toBe(85);
    expect(mapImagePhaseProgress("Local image: 100% ready")).toBe(100);
    expect(mapImagePhaseProgress("no percent here")).toBe(70);
  });
});

describe("offlinePack — initial state & last-downloaded label", () => {
  it("initial state when never downloaded shows phase=idle, 0%", () => {
    const s = initialOfflinePackState(null);
    expect(s.phase).toBe("idle");
    expect(s.overallPct).toBe(0);
    expect(s.completedAt).toBeNull();
  });

  it("initial state when previously downloaded shows phase=done, 100%", () => {
    const s = initialOfflinePackState(1714000000000);
    expect(s.phase).toBe("done");
    expect(s.overallPct).toBe(100);
    expect(s.completedAt).toBe(1714000000000);
  });

  it("formatted timestamp is a non-empty Hebrew string for valid input", () => {
    expect(formatOfflinePackCompletedAt(null)).toBe("");
    expect(formatOfflinePackCompletedAt(0)).toBe("");
    expect(formatOfflinePackCompletedAt(NaN)).toBe("");
    const s = formatOfflinePackCompletedAt(Date.now());
    expect(s.startsWith("הורדה אחרונה")).toBe(true);
  });
});

describe("offlinePack — localStorage round-trip", () => {
  it("read returns null when the key is missing", () => {
    expect(readOfflinePackCompletedAt()).toBeNull();
  });

  it("write+read preserve the timestamp", () => {
    writeOfflinePackCompletedAt(1714123456789);
    expect(readOfflinePackCompletedAt()).toBe(1714123456789);
    expect(localStorage.getItem(OFFLINE_PACK_LOCAL_STORAGE_KEY)).toBe("1714123456789");
  });

  it("clear removes the key", () => {
    writeOfflinePackCompletedAt(123);
    clearOfflinePackCompletedAt();
    expect(readOfflinePackCompletedAt()).toBeNull();
  });

  it("read returns null for a non-numeric stored value", () => {
    localStorage.setItem(OFFLINE_PACK_LOCAL_STORAGE_KEY, "garbage");
    expect(readOfflinePackCompletedAt()).toBeNull();
  });
});
