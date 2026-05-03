import { describe, expect, it } from "vitest";
import { sdTurboWorkerLoadOptions } from "./localImageGen";

describe("localImageGen / web-txt2img Worker bridge", () => {
  it("sdTurboWorkerLoadOptions is structuredClone-safe (no functions in payload)", () => {
    for (const useGpu of [true, false]) {
      const opts = sdTurboWorkerLoadOptions(useGpu);
      expect(() => structuredClone(opts)).not.toThrow();
      expect(opts).not.toHaveProperty("onProgress");
      expect(typeof (opts as { onProgress?: unknown }).onProgress).toBe("undefined");
    }
  });

  it("prefers WebGPU first only when flag is true", () => {
    expect(sdTurboWorkerLoadOptions(true).backendPreference[0]).toBe("webgpu");
    expect(sdTurboWorkerLoadOptions(false).backendPreference).toEqual(["wasm"]);
  });
});
