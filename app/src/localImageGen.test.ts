import { afterEach, describe, expect, it, vi } from "vitest";
import {
  __setImageClientFactoryForTests,
  buildGenerateParams,
  buildLoadOptions,
  ensureSdTurboLoaded,
  generateSdTurboPng,
  isWebGpuStateError,
  terminateLocalImageWorker,
  type GenProgressLike,
  type ImageWorkerClientLike,
  type LoadProgressLike,
} from "./localImageGen";

/**
 * The whole "image not generating" production bug was a `DataCloneError`:
 * we accidentally put an `onProgress` function inside the options object
 * that web-txt2img postMessage's to its worker. Functions can't be
 * structured-cloned, so the call rejected before the worker ever started.
 *
 * These tests lock that contract:
 *   - load options + generate params are pure data (no functions, no AbortSignal)
 *   - structuredClone() of those payloads succeeds
 *   - the third-arg onProgress callback is what receives status updates
 *   - end-to-end flow returns a blob: URL via URL.createObjectURL
 */

afterEach(() => {
  __setImageClientFactoryForTests(null);
  terminateLocalImageWorker();
  vi.restoreAllMocks();
});

const mockNavigator = (gpu: boolean) => {
  // Polyfill structuredClone for older Node test runners (defensive).
  if (typeof globalThis.structuredClone !== "function") {
    globalThis.structuredClone = ((v: unknown) =>
      JSON.parse(JSON.stringify(v))) as typeof globalThis.structuredClone;
  }
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: gpu
      ? {
          gpu: {
            requestAdapter: vi.fn().mockResolvedValue({ name: "fake" }),
          },
        }
      : { gpu: undefined },
  });
};

const mockUrlCreators = () => {
  const urls: string[] = [];
  Object.defineProperty(globalThis, "URL", {
    configurable: true,
    value: {
      ...globalThis.URL,
      createObjectURL: (b: Blob) => {
        const u = `blob:mock-${urls.length}-${b.size}`;
        urls.push(u);
        return u;
      },
      revokeObjectURL: () => {},
    } as unknown as typeof URL,
  });
  return urls;
};

type LoadCall = {
  model: string;
  options: unknown;
  onProgress?: (p: LoadProgressLike) => void;
};

type GenCall = {
  params: unknown;
  onProgress?: (e: GenProgressLike) => void;
};

const buildClient = (overrides: Partial<ImageWorkerClientLike> = {}): {
  client: ImageWorkerClientLike;
  loadCalls: LoadCall[];
  genCalls: GenCall[];
} => {
  const loadCalls: LoadCall[] = [];
  const genCalls: GenCall[] = [];
  const client: ImageWorkerClientLike = {
    detect: async () => ({ webgpu: true, wasm: true, shaderF16: false }),
    load: async (model, options, onProgress) => {
      loadCalls.push({ model, options, onProgress });
      structuredClone(options);
      onProgress?.({ phase: "loading", pct: 50, message: "downloading", asset: "unet" });
      return { ok: true, backendUsed: "wasm" };
    },
    generate: (params, onProgress) => {
      genCalls.push({ params, onProgress });
      structuredClone(params);
      onProgress?.({ phase: "denoising", pct: 0.42 });
      const blob = new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: "image/png" });
      return {
        id: "mock-gen-1",
        promise: Promise.resolve({ type: "result", ok: true, blob, timeMs: 12 }),
        abort: async () => {},
      };
    },
    terminate: () => {},
    ...overrides,
  };
  return { client, loadCalls, genCalls };
};

describe("localImageGen — payload contract", () => {
  it("buildLoadOptions is structured-clone safe (no functions)", () => {
    const opts = buildLoadOptions(true);
    expect(() => structuredClone(opts)).not.toThrow();
    for (const v of Object.values(opts)) expect(typeof v).not.toBe("function");
    expect(opts.backendPreference).toEqual(["webgpu", "wasm"]);
    expect(buildLoadOptions(false).backendPreference).toEqual(["wasm"]);
  });

  it("buildGenerateParams is structured-clone safe and includes the prompt verbatim", () => {
    const params = buildGenerateParams('a "fluffy" cat 🐈');
    expect(() => structuredClone(params)).not.toThrow();
    for (const v of Object.values(params)) expect(typeof v).not.toBe("function");
    expect(params.model).toBe("sd-turbo");
    expect(params.prompt).toBe('a "fluffy" cat 🐈');
    expect(params.width).toBe(512);
    expect(params.height).toBe(512);
  });
});

describe("localImageGen — load flow", () => {
  it("calls load(model, options, onProgress) with onProgress as 3rd arg, never inside options", async () => {
    mockNavigator(true);
    const { client, loadCalls } = buildClient();
    __setImageClientFactoryForTests(() => client);

    const statuses: string[] = [];
    const ok = await ensureSdTurboLoaded((s) => statuses.push(s));

    expect(ok).toBe(true);
    expect(loadCalls).toHaveLength(1);
    const call = loadCalls[0];
    expect(call.model).toBe("sd-turbo");
    expect(typeof call.onProgress).toBe("function");
    // The killer assertion: NO function snuck into the cloned options object.
    expect(() => structuredClone(call.options)).not.toThrow();
    const opts = call.options as Record<string, unknown>;
    for (const v of Object.values(opts)) expect(typeof v).not.toBe("function");
    expect(statuses.some((s) => s.includes("50%") && s.includes("downloading"))).toBe(true);
    expect(statuses).toContain("Local image: SD-Turbo ready");
  });

  it("falls back to WASM-only when no WebGPU adapter is available", async () => {
    mockNavigator(false);
    const { client, loadCalls } = buildClient();
    __setImageClientFactoryForTests(() => client);

    await ensureSdTurboLoaded(() => {});
    expect((loadCalls[0].options as { backendPreference: string[] }).backendPreference).toEqual(["wasm"]);
  });

  it("returns false and reports message when worker reports load failure", async () => {
    mockNavigator(true);
    const { client } = buildClient({
      load: async () => ({ ok: false, reason: "backend_unavailable", message: "ORT init failed" }),
    });
    __setImageClientFactoryForTests(() => client);

    const statuses: string[] = [];
    const ok = await ensureSdTurboLoaded((s) => statuses.push(s));
    expect(ok).toBe(false);
    expect(statuses.some((s) => s.includes("ORT init failed"))).toBe(true);
  });

  it("does not double-load on concurrent calls (single-flight)", async () => {
    mockNavigator(true);
    let calls = 0;
    const slowClient: ImageWorkerClientLike = {
      detect: async () => ({ webgpu: true, wasm: true, shaderF16: false }),
      load: async () => {
        calls += 1;
        await new Promise((r) => setTimeout(r, 5));
        return { ok: true, backendUsed: "wasm" };
      },
      generate: () => ({
        id: "x",
        promise: Promise.resolve({ ok: false }),
        abort: async () => {},
      }),
      terminate: () => {},
    };
    __setImageClientFactoryForTests(() => slowClient);

    const [a, b] = await Promise.all([ensureSdTurboLoaded(() => {}), ensureSdTurboLoaded(() => {})]);
    expect(a).toBe(true);
    expect(b).toBe(true);
    expect(calls).toBe(1);
  });
});

describe("localImageGen — generate flow", () => {
  it("returns blob: URL on success and reports progress without sending functions to the worker", async () => {
    mockNavigator(true);
    const urls = mockUrlCreators();
    const { client, genCalls } = buildClient();
    __setImageClientFactoryForTests(() => client);

    const statuses: string[] = [];
    const result = await generateSdTurboPng("a cat on a windowsill", (s) => statuses.push(s));
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.objectUrl.startsWith("blob:")).toBe(true);
    expect(urls.length).toBe(1);

    expect(genCalls).toHaveLength(1);
    const call = genCalls[0];
    expect(typeof call.onProgress).toBe("function");
    expect(() => structuredClone(call.params)).not.toThrow();
    const params = call.params as Record<string, unknown>;
    for (const v of Object.values(params)) expect(typeof v).not.toBe("function");
    expect((params as { prompt: string }).prompt).toBe("a cat on a windowsill");

    expect(statuses).toContain("Local image: done");
    expect(statuses.some((s) => s.includes("denoising"))).toBe(true);
  });

  it("returns ok=false with the worker-reported reason when generation fails", async () => {
    mockNavigator(true);
    const { client } = buildClient({
      generate: () => ({
        id: "x",
        promise: Promise.resolve({ type: "result", ok: false, reason: "internal_error", message: "shader compile failed" }),
        abort: async () => {},
      }),
    });
    __setImageClientFactoryForTests(() => client);

    const out = await generateSdTurboPng("blue moon", () => {});
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.message).toBe("shader compile failed");
  });

  it("returns ok=false when the worker promise rejects (e.g. DataCloneError) and aborts safely", async () => {
    mockNavigator(true);
    const abort = vi.fn(async () => {});
    const { client } = buildClient({
      generate: () => ({
        id: "x",
        promise: Promise.reject(new Error("Failed to execute 'postMessage' on 'Worker'")),
        abort,
      }),
    });
    __setImageClientFactoryForTests(() => client);

    const out = await generateSdTurboPng("starlight", () => {});
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.message).toContain("postMessage");
    expect(abort).toHaveBeenCalled();
  });

  it("returns ok=false when SD-Turbo could not be loaded", async () => {
    mockNavigator(true);
    const { client } = buildClient({ detect: async () => ({ webgpu: false, wasm: false, shaderF16: false }) });
    __setImageClientFactoryForTests(() => client);

    const out = await generateSdTurboPng("anything", () => {});
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.message).toBe("SD-Turbo not loaded");
  });
});

describe("localImageGen — isWebGpuStateError", () => {
  it("matches the production GPU-state error families", () => {
    expect(isWebGpuStateError("Cannot read properties of undefined (reading 'destroy')")).toBe(true);
    expect(isWebGpuStateError("A valid external Instance reference no longer exists.")).toBe(true);
    expect(isWebGpuStateError("GPU device was lost: device_lost")).toBe(true);
    expect(isWebGpuStateError("Aborted(both async and sync fetching of the wasm failed)")).toBe(true);
  });

  it("does not match unrelated errors", () => {
    expect(isWebGpuStateError(null)).toBe(false);
    expect(isWebGpuStateError("Failed to fetch")).toBe(false);
    expect(isWebGpuStateError("404 Not Found")).toBe(false);
    expect(isWebGpuStateError("")).toBe(false);
  });
});

describe("localImageGen — forceBackend wasm fallback", () => {
  it("never asks for a WebGPU adapter when forceBackend='wasm'", async () => {
    const requestAdapter = vi.fn();
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: { gpu: { requestAdapter } },
    });
    if (typeof globalThis.structuredClone !== "function") {
      globalThis.structuredClone = ((v: unknown) => JSON.parse(JSON.stringify(v))) as typeof globalThis.structuredClone;
    }

    const calls: { backend: string[] }[] = [];
    const client: ImageWorkerClientLike = {
      detect: async () => ({ webgpu: true, wasm: true, shaderF16: false }),
      load: async (_model, options) => {
        calls.push({ backend: (options as { backendPreference: string[] }).backendPreference });
        return { ok: true, backendUsed: "wasm" };
      },
      generate: () => ({ id: "x", promise: Promise.resolve({ ok: false }), abort: async () => {} }),
      terminate: () => {},
    };
    __setImageClientFactoryForTests(() => client);

    const ok = await ensureSdTurboLoaded(() => {}, { forceBackend: "wasm" });
    expect(ok).toBe(true);
    expect(calls[0].backend).toEqual(["wasm"]);
    expect(requestAdapter).not.toHaveBeenCalled();
  });
});
