/**
 * Fallback when `web-txt2img` is not installed (default GROVEEMODEL checkout).
 * Vite aliases `web-txt2img` → this file when the package is absent.
 */

const NOT_INSTALLED = "web-txt2img is not installed — use Pollinations (FLUX) or npm install web-txt2img";

class UnavailableTxt2ImgClient {
  async detect() {
    return { wasm: false, webgpu: false, shaderF16: false };
  }

  async load() {
    return { ok: false, message: NOT_INSTALLED };
  }

  generate() {
    return {
      id: "unavailable",
      promise: Promise.resolve({ ok: false, message: NOT_INSTALLED }),
      abort: async () => {},
    };
  }

  terminate() {}
}

export class Txt2ImgWorkerClient {
  static createDefault(): UnavailableTxt2ImgClient {
    return new UnavailableTxt2ImgClient();
  }
}
