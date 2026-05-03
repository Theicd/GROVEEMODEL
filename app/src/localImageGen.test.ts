import { describe, expect, it } from "vitest";
import { webTxt2ImgLoadOptions, type ImageGenModelId } from "./localImageGen";

describe("localImageGen / web-txt2img Worker bridge", () => {
  it("webTxt2ImgLoadOptions is structuredClone-safe for each model (no functions in payload)", () => {
    const models: ImageGenModelId[] = ["sd-turbo", "janus-pro-1b"];
    for (const modelId of models) {
      for (const useGpu of [true, false]) {
        const opts = webTxt2ImgLoadOptions(modelId, useGpu);
        expect(() => structuredClone(opts)).not.toThrow();
        expect(opts).not.toHaveProperty("onProgress");
        expect(typeof (opts as { onProgress?: unknown }).onProgress).toBe("undefined");
      }
    }
  });

  it("sd-turbo respects WebGPU-first vs WASM-only; Janus forces WebGPU only", () => {
    expect(webTxt2ImgLoadOptions("sd-turbo", true).backendPreference).toEqual(["webgpu", "wasm"]);
    expect(webTxt2ImgLoadOptions("sd-turbo", false).backendPreference).toEqual(["wasm"]);
    expect(webTxt2ImgLoadOptions("janus-pro-1b", true).backendPreference).toEqual(["webgpu"]);
    expect(webTxt2ImgLoadOptions("janus-pro-1b", false).backendPreference).toEqual(["webgpu"]);
  });
});
