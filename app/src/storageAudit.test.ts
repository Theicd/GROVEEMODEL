import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  auditAppStorage,
  isAuditEmpty,
  summarizeStorageHeader,
  summarizeStorageInventory,
  type StorageAudit,
} from "./storageAudit";

const installEnv = (env: {
  estimate?: () => Promise<{ usage?: number; quota?: number }>;
  persisted?: () => Promise<boolean>;
  cacheNames?: string[];
  cacheEntryCounts?: Record<string, number>;
  idbNames?: string[];
  opfsEntries?: string[];
  swRegistrations?: number;
}) => {
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      storage: {
        estimate: env.estimate,
        persisted: env.persisted,
        getDirectory: env.opfsEntries
          ? async () => ({
              async *keys() {
                for (const k of env.opfsEntries!) yield k;
              },
            })
          : undefined,
      },
      serviceWorker:
        env.swRegistrations !== undefined
          ? { getRegistrations: async () => Array(env.swRegistrations).fill({}) }
          : undefined,
    },
  });
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: env.cacheNames
      ? {
          keys: async () => env.cacheNames!,
          open: async (name: string) => ({
            keys: async () => Array(env.cacheEntryCounts?.[name] ?? 0).fill({}),
          }),
        }
      : undefined,
  });
  Object.defineProperty(globalThis, "indexedDB", {
    configurable: true,
    value: env.idbNames
      ? { databases: async () => env.idbNames!.map((name) => ({ name })) }
      : { databases: undefined },
  });
};

beforeEach(() => {
  installEnv({});
});

afterEach(() => {
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: undefined });
  Object.defineProperty(globalThis, "caches", { configurable: true, value: undefined });
  Object.defineProperty(globalThis, "indexedDB", { configurable: true, value: undefined });
});

describe("storageAudit — auditAppStorage", () => {
  it("aggregates usage, persisted, caches, IDB, OPFS and service worker counts", async () => {
    installEnv({
      estimate: async () => ({ usage: 3.5 * 1024 ** 3, quota: 12 * 1024 ** 3 }),
      persisted: async () => true,
      cacheNames: ["transformers-cache", "web-txt2img-v1"],
      cacheEntryCounts: { "transformers-cache": 17, "web-txt2img-v1": 5 },
      idbNames: ["onnx-cache"],
      opfsEntries: ["models", "ort-wasm"],
      swRegistrations: 1,
    });

    const a = await auditAppStorage();
    expect(a.usageBytes).toBeCloseTo(3.5 * 1024 ** 3, -3);
    expect(a.quotaBytes).toBeCloseTo(12 * 1024 ** 3, -3);
    expect(a.persisted).toBe(true);
    expect(a.caches.map((c) => c.name)).toEqual(["transformers-cache", "web-txt2img-v1"]);
    expect(a.caches.find((c) => c.name === "transformers-cache")?.entryCount).toBe(17);
    expect(a.idbNames).toEqual(["onnx-cache"]);
    expect(a.opfsEntries).toEqual(["models", "ort-wasm"]);
    expect(a.serviceWorkerCount).toBe(1);
    expect(a.errors).toEqual([]);
  });

  it("collects errors instead of throwing when an API explodes", async () => {
    installEnv({
      estimate: async () => {
        throw new Error("private mode");
      },
    });
    const a = await auditAppStorage();
    expect(a.usageBytes).toBe(-1);
    expect(a.errors.some((e) => e.includes("estimate"))).toBe(true);
  });

  it("notes when indexedDB.databases() is not supported", async () => {
    installEnv({});
    const a = await auditAppStorage();
    expect(a.errors.some((e) => /indexedDB\.databases/.test(e))).toBe(true);
  });
});

describe("storageAudit — summary helpers", () => {
  const baseAudit: StorageAudit = {
    usageBytes: 3.5 * 1024 ** 3,
    quotaBytes: 12 * 1024 ** 3,
    persisted: true,
    caches: [{ name: "transformers-cache", entryCount: 17 }],
    idbNames: ["a", "b"],
    opfsEntries: ["models"],
    serviceWorkerCount: 0,
    errors: [],
  };

  it("header shows used/total + Hebrew persisted label", () => {
    expect(summarizeStorageHeader(baseAudit)).toBe("3.50 GB / 12.00 GB · persisted: כן");
  });

  it("header marks persisted=no when not persistent", () => {
    expect(summarizeStorageHeader({ ...baseAudit, persisted: false })).toContain("persisted: לא");
  });

  it("inventory shows category counts in a single line", () => {
    expect(summarizeStorageInventory(baseAudit)).toBe("1 caches · 2 IDB · 1 OPFS · 0 SW");
  });

  it("isAuditEmpty true only when nothing is stored", () => {
    expect(isAuditEmpty(baseAudit)).toBe(false);
    expect(
      isAuditEmpty({
        ...baseAudit,
        caches: [],
        idbNames: [],
        opfsEntries: [],
        serviceWorkerCount: 0,
        usageBytes: 0,
      }),
    ).toBe(true);
  });
});
