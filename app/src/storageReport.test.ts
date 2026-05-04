import { afterEach, describe, expect, it, vi } from "vitest";
import { formatBytes, requestPersistentStorage } from "./storageReport";

describe("storageReport.formatBytes", () => {
  it("formats common units with sensible precision", () => {
    expect(formatBytes(0)).toBe("0 MB");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5 MB");
    expect(formatBytes(2.5 * 1024 * 1024 * 1024)).toBe("2.50 GB");
  });

  it("returns '?' for invalid input (so the UI never shows NaN)", () => {
    expect(formatBytes(Number.NaN)).toBe("?");
    expect(formatBytes(-1)).toBe("?");
    expect(formatBytes(Infinity)).toBe("?");
  });
});

const setNavigatorStorage = (storage: unknown) => {
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { storage },
  });
};

describe("storageReport.requestPersistentStorage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns supported=false when StorageManager.persist is missing", async () => {
    setNavigatorStorage({});
    const r = await requestPersistentStorage();
    expect(r.supported).toBe(false);
    expect(r.persistent).toBe(false);
  });

  it("does not call persist() when already persistent", async () => {
    const persist = vi.fn().mockResolvedValue(false);
    setNavigatorStorage({ persisted: vi.fn().mockResolvedValue(true), persist });
    const r = await requestPersistentStorage();
    expect(r.supported).toBe(true);
    expect(r.persistent).toBe(true);
    expect(persist).not.toHaveBeenCalled();
  });

  it("calls persist() when not yet persistent and reports the result", async () => {
    const persist = vi.fn().mockResolvedValue(true);
    setNavigatorStorage({ persisted: vi.fn().mockResolvedValue(false), persist });
    const r = await requestPersistentStorage();
    expect(r.persistent).toBe(true);
    expect(persist).toHaveBeenCalled();
  });

  it("reports denied when persist() returns false (Brave / private)", async () => {
    setNavigatorStorage({
      persisted: vi.fn().mockResolvedValue(false),
      persist: vi.fn().mockResolvedValue(false),
    });
    const r = await requestPersistentStorage();
    expect(r.persistent).toBe(false);
    expect(r.reason).toBe("denied");
  });

  it("never throws when the API throws — surfaces the error in `reason`", async () => {
    setNavigatorStorage({
      persisted: vi.fn().mockRejectedValue(new Error("boom")),
      persist: vi.fn(),
    });
    const r = await requestPersistentStorage();
    expect(r.persistent).toBe(false);
    expect(r.reason).toBe("boom");
  });
});
