declare module "web-txt2img" {
  export type LoadProgress = {
    phase?: string;
    message?: string;
    pct?: number;
    asset?: string;
  };

  export class Txt2ImgWorkerClient {
    static createDefault(): Txt2ImgWorkerClient;
    detect(): Promise<{ webgpu?: boolean; wasm?: boolean; shaderF16?: boolean }>;
    load(
      model: string,
      options: { backendPreference: ("webgpu" | "wasm")[] },
      onProgress?: (p: LoadProgress) => void,
    ): Promise<{ ok: boolean; reason?: string; message?: string } | unknown>;
    generate(
      prompt: string,
      options?: Record<string, unknown>,
      onProgress?: (p: { phase?: string; pct?: number }) => void,
    ): Promise<{ ok: boolean; image?: string; reason?: string; message?: string } | unknown>;
    dispose(): void;
  }
}
